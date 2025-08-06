from transformers.tokenization_utils_base import PaddingStrategy, TensorType, TruncationStrategy
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationMixin
try:
    from transformers import FuyuProcessor, FuyuForCausalLM, FuyuProcessor, FuyuPreTrainedModel, FuyuConfig
    from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor, FuyuBatchFeature
except ImportError:
    print("Fuyu is not found, use new version of transformers")

try:
    from transformers.models.instructblip.modeling_instructblip import InstructBlipQFormerModel
    from transformers.models.instructblip.configuration_instructblip import InstructBlipQFormerConfig
    from models.qformer_moe import build_moe_qformer_from_config

except ImportError:
    print("InstructBlip is not found, use new version of transformers")

try:
    from transformers import MistralForCausalLM, MistralConfig
except ImportError:
    print("Mistral is not found, use new version of transformers")
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine import MinkowskiToDenseTensor
    from models.minkunet import MinkUNet34C
except ImportError:
    print("MinkowskiEngine is not installed, please install if you want to use MinkowskiNetTokenizer")
    pass

# from models.backbone_module import Pointnet2Backbone
# from models.voting_module import VotingModule
# from models.proposal_module import ProposalModule
try:
    from models.pnpp import PointNetPP
    from models.detector_Vote2Cap_DETR.detector import detector
except ImportError:
    print("PointNet++ is not installed, please install if you want to use PointNetPPTokenizer")
    pass


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import PeftModel
from icecream import ic
from typing import Optional, Tuple, Union, List
import os
from dataclasses import dataclass, asdict
from fuyu_utils import batch_calculate_reinforce_reward_labels, loss_reinforce, print_once, AverageMeter
# from lib.dataset import DC
from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
import numpy as np
import math
# from accelerate import get_accelerator
from fuyu_align_utils import (
    calculate_in_view_objects,
    calculate_related_objects,
)

from transformers.models.deformable_detr import DeformableDetrConfig
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)

# from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

# from LL3DA_main.models.detector_Vote2Cap_DETR.detector import detector
from collections import namedtuple
import inspect

import logging


logger = logging.getLogger(__name__)
average_meter = AverageMeter(report_period=10, print_fn=print)

def trim_objects(out, coords_mask, predicted_bbox_corners):
    # according to coords_mask, trim the objects
    # out ~ [B, N, H]
    # coords_mask ~ [B, N]
    # predicted_bbox_corners ~ [B, N, 8, 3]
    print_once("Enable trim_objects for less tokens")

    coords_mask = coords_mask.bool()
    max_num_objects = coords_mask.sum(dim=1).max().item()
    new_out = torch.zeros(out.shape[0], max_num_objects, out.shape[-1], device=out.device)
    new_coords_mask = torch.zeros(out.shape[0], max_num_objects, dtype=torch.bool, device=out.device)
    if predicted_bbox_corners is not None:
        new_predicted_bbox_corners = torch.zeros(out.shape[0], max_num_objects, predicted_bbox_corners.shape[2], predicted_bbox_corners.shape[3], device=out.device)

    for i in range(out.shape[0]):
        new_out[i, :coords_mask[i].sum()] = out[i, coords_mask[i]]
        new_coords_mask[i, :coords_mask[i].sum()] = True
        if predicted_bbox_corners is not None:
            new_predicted_bbox_corners[i, :coords_mask[i].sum()] = predicted_bbox_corners[i, coords_mask[i]]

    return new_out, new_coords_mask, new_predicted_bbox_corners

def choose_related_objects(out, coords_mask, relative_object_mask, kept_objects=60):
    # first, choose objects that are related to the task
    # then, random select objects to keep, until a fixed number of objects are kept
    # out ~ [B, N, H], can consider that it is trimmed - only first N objects are valid
    # coords_mask ~ [B, N]
    # relative_object_mask ~ [B, N], 0 or (min_iou-1) for related objects
    relative_object_mask = relative_object_mask > 0
    relative_object_mask = relative_object_mask & coords_mask
    # choose related objects
    new_out = torch.zeros(out.shape[0], kept_objects, out.shape[-1], device=out.device)
    new_coords_mask = torch.zeros(out.shape[0], kept_objects, dtype=torch.bool, device=out.device)
    for i in range(out.shape[0]):
        # new_out[i, :relative_object_mask[i].sum()] = out[i, relative_object_mask[i]]
        new_out_row = out[i, relative_object_mask[i]][:kept_objects]
        new_out[i, :new_out_row.shape[0]] = new_out_row
        new_coords_mask[i, :new_out_row.shape[0]] = True

    # random select objects to keep
    for i in range(out.shape[0]):
        already_kept = relative_object_mask[i].sum()
        if already_kept < kept_objects:
            # select random objects other than related objects
            # in first N objects, already_kept objects are related objects, so we need to select from N - already_kept objects
            random_objects = torch.randperm(coords_mask[i].sum() - already_kept)[:kept_objects - already_kept]
            # irrelative_but_valid = coords_mask[i] & ~relative_object_mask[i]
            new_out_row = out[i, (coords_mask[i] & ~relative_object_mask[i])][random_objects]
            new_out[i, already_kept:already_kept+new_out_row.shape[0]] = new_out_row


    return new_out, new_coords_mask



class MyObjectDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(f"Attribute {name} not found")
    
    def __setattr__(self, name, value):
        self[name] = value


class Vote2CapDETRTokenizer(nn.Module):
    def __init__(self, config, dataset_config=DC):
        super().__init__()
        self.config = config
        self.dataset_config = dataset_config
        self.detector = detector(config, dataset_config)
        self.return_type = config.return_type

        logger.info(f"Bulding Vote2Cap-DETR tokenizer, return_type={self.return_type}")
        if self.return_type == "enc_features":
            self.out_channels = 256 + 3
        elif self.return_type == "box_features":
            self.out_channels = 256 + 1 + (self.detector.dataset_config.num_semcls) + 3 + 3 # box_features + objectness + sem_cls + center + size

    def forward(self, inputs: Tuple):
        inputs = {
            "point_clouds": inputs[0],
            "point_cloud_dims_min": inputs[1],
            "point_cloud_dims_max": inputs[2],
        }

        with torch.cuda.amp.autocast(enabled=False):
            out = self.detector(inputs, is_eval=True)
        if self.return_type == "enc_features":
            enc_features = out['enc_features']
            enc_xyz = out['enc_xyz']
            enc_features = torch.cat((enc_features, enc_xyz), dim=-1)

            enc_mask = torch.ones(*enc_features.shape[:-1], dtype=torch.bool, device=enc_features.device)
            return enc_features, enc_mask
        elif self.return_type == "box_features":
            box_features = out['prop_features'] #nlayers x batch x nqueries x channel
            box_features = box_features[-1] # last layer, [B, N, C]
            objectness_prob = out['objectness_prob'] # [B, N]
            box_mask = objectness_prob > 0.5

            # aggregate other box predictions
            sem_cls_logits = out['sem_cls_logits'] # [B, N, N_cls] # addded 1 class for background/not-object
            center = out['center_normalized'] # [B, N, 3]
            size = out['size_normalized'] # [B, N, 3]
            aggregate_box_features = torch.cat((box_features, sem_cls_logits, center, size), dim=-1) # [B, N, C+N_cls+3+3]

            bbox_corners = out['box_corners'] # [B, N, 8, 3], for frame caption check

            return aggregate_box_features, box_mask, bbox_corners
        else:
            raise ValueError(f"Unknown return type: {self.return_type}")
            
    
    def load_pretrained(self, path=None):
        if path is None:
            path = self.config.vote2cap_detr_path
        # load pretrained Vote2Cap-DETR
        print(f"Loading pretrained Vote2Cap-DETR from {path}")
        state_dict = torch.load(path, map_location=torch.device('cpu'))['model']
        message = self.load_state_dict(state_dict, strict=False)
        print(message)
        
        if self.config.freeze_vote2cap_detr:
            print("Freezing Vote2Cap-DETR")
            self.detector.eval()
            for p in self.detector.parameters():
                p.requires_grad = False


class PointCloudTokenizer(nn.Module):
    """
    Point cloud encoder
    """
    def __init__(self, config):
        super().__init__()
        self.config = config


class MinkowskiNetTokenizer(PointCloudTokenizer):
    def __init__(self, config):
        super().__init__(config)

        self.in_channels = self.config.get("in_channels", 3)
        self.spatial_patch_size = self.config.get("spatial_patch_size", 24)
        self.pooling_method = self.config.get("pooling_method", "max")
        self.mnet_path = self.config.get("mnet_path", "/scratch/generalvision/mowentao/ScanQA/weights.pth")
        self.freeze_mnet = self.config.get("freeze_mnet", True)

        self.mnet = MinkUNet34C(self.in_channels, 20)
        self.out_channels_mnet = self.mnet.PLANES[7] * self.mnet.BLOCK.expansion + 3
        print(f"MinkUNet out_channels: {self.out_channels_mnet}")
        
        if self.pooling_method == "max":
            self.mnet_pool = ME.MinkowskiMaxPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)
        elif self.pooling_method == "avg":
            self.mnet_pool = ME.MinkowskiAvgPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)
        elif self.pooling_method == "sum":
            self.mnet_pool = ME.MinkowskiSumPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)

    def forward(self, mnet_inputs):
        coords, feats, point_labels = mnet_inputs
        # MinkowskiNet/3D scene encoding
        # diable autocast
        with torch.cuda.amp.autocast(enabled=False): # DO NOT use autocast here, since MinkowskiNet is float32
            # assert mnet_inputs is not None, "mnet_inputs must be provided for MinkowskiNet"
            coords, feats, point_labels = mnet_inputs
            # MinkowskiNet/3D scene encoding
            # FIXME: shall we skip if past_key_values is not None?
            out, pred = self.mnet(ME.SparseTensor(feats.float(), coords))
            # ic(out.shape)
            out = self.mnet_pool(out)
            # ic(out.shape)

            coords_batch, feats_batch = out.decomposed_coordinates_and_features
            # how to deal with inconsistent point/voxel numbers?
            # pad to the same number of points
            
        coords_batch = pad_sequence([coords for coords in coords_batch], batch_first=True, padding_value=0) # [B, N, 3]
        feats_batch = pad_sequence([feats for feats in feats_batch], batch_first=True, padding_value=0) # [B, N, C]
        # ic(coords_batch.shape, feats_batch.shape)
        coords_mask = torch.logical_not(torch.all(coords_batch == 0, dim=-1)) # [B, N]

        out = torch.cat((coords_batch, feats_batch), dim=-1) # [B, N, 3+C]
        return out, coords_mask
    

    def load_pretrained(self, path=None):
        if path is None:
            path = self.mnet_path
        
        # load pretrained MinkowskiNet
        print(f"Loading pretrained MinkowskiNet from {self.config.mnet_path}, freeze_mnet={self.config.freeze_mnet}")
        self.mnet = self.mnet.float() # MinkowskiNet is float32, not supporting half-precision
        model_dict = torch.load(self.config.mnet_path, map_location=torch.device('cpu'))
        self.mnet.load_state_dict(model_dict)
        if self.freeze_mnet:
            self.mnet.eval()
            for p in self.mnet.parameters():
                p.requires_grad = False

@dataclass
class PointNetConfig():
    num_proposal: int = 256
    vote_factor: int = 1
    sampling: str = "vote_fps"
    seed_feat_dim: int = 256
    proposal_size: int = 128
    pointnet_width: int = 1
    pointnet_depth: int = 2
    vote_radius: float= 0.3
    vote_nsample: int = 16
                
# from data.scannet.model_util_scannet import ScannetQADatasetConfig
# DC = ScannetQADatasetConfig()
    
class PointNetPPTokenizer(PointCloudTokenizer):
    def __init__(self, config):
        super().__init__(config)

        merged_config = {**asdict(PointNetConfig())}
        for k, v in merged_config.items():
            if getattr(config, k, None) is not None:
                merged_config[k] = getattr(config, k)

        logger.info(f"merged_config: {merged_config}")

        self.pnpp = PointNetPP(
            input_feature_dim=config.in_channels,            
            num_object_class=DC.num_class, 
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            **merged_config,
        )

        self.out_channels = merged_config["proposal_size"]

    def forward(self, mnet_inputs):
        # PointNet++/3D scene encoding
        # print(mnet_inputs.dtype, mnet_inputs.device)
        with torch.cuda.amp.autocast(enabled=False):
            out = self.pnpp(mnet_inputs.float())
        return out
    
    def load_pretrained(self, path=None):
        self.pnpp.load_pretrained(path if path is not None else self.config.pnpp_path)
        if self.config.freeze_pnpp:
            self.pnpp.eval()
            for p in self.pnpp.parameters():
                p.requires_grad = False

class FrozenPointCloudTokenizer(PointCloudTokenizer):
    def __init__(self):
        super().__init__(None)

    def forward(self, mnet_inputs):
        return mnet_inputs

    def load_pretrained(self, path=None):
        pass

class LinearEncoders(nn.Module):
    """
    class to contain linear_3d and linear_focus_bbox
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        out_channels_mnet = config.out_channels_mnet
        adapter_type = config.adapter_type
        if adapter_type == "ffn":
            self.linear_3d = nn.Sequential(
                nn.LayerNorm(out_channels_mnet),
                nn.Linear(out_channels_mnet, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )
            self.linear_focus_bbox = nn.Sequential(
                nn.Linear(6, self.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )
        elif adapter_type == "linear":
            # self.linear_3d = nn.Linear(out_channels_mnet + 3, self.config.hidden_size)
            self.linear_3d = nn.Linear(out_channels_mnet, self.config.hidden_size)
            self.linear_focus_bbox = nn.Linear(6, self.config.hidden_size)
        elif adapter_type == "deformable-detr":
            raise NotImplementedError("Deformable DETR adapter is not implemented yet")
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

    def forward(self, scene_embeds, focus_bbox=None):
        return (
            self.linear_3d(scene_embeds),
            self.linear_focus_bbox(focus_bbox) if focus_bbox is not None else None
        )
    

class PCTokenizerAdapterMixin:
    def _init_pc_tokenizer(self):
        pc_tokenizer_type = self.config.pc_tokenizer_type
        if pc_tokenizer_type == "minkowski":
            self.pc_tokenizer = MinkowskiNetTokenizer(self.config)
            out_channels_mnet = self.pc_tokenizer.out_channels_mnet
        elif pc_tokenizer_type == "pointnet++":
            self.pc_tokenizer = PointNetPPTokenizer(self.config)
            out_channels_mnet = self.pc_tokenizer.out_channels
        elif pc_tokenizer_type == "frozen":
            # out_channels_mnet = self.config.get("out_channels", PointNetConfig().proposal_size)
            # out_channels_mnet = getattr(self.config, "out_channels", PointNetConfig().proposal_size)
            self.pc_tokenizer = FrozenPointCloudTokenizer()
            out_channels_mnet = self.config.frozen_in_channels
        elif pc_tokenizer_type == "vote2cap-detr":
            # self.config.return_type = self.config.get("vote2cap_return_type", "enc_features")
            self.config.return_type = getattr(self.config, "vote2cap_return_type", "enc_features")
            self.pc_tokenizer = Vote2CapDETRTokenizer(self.config, dataset_config=DC)
            out_channels_mnet = self.pc_tokenizer.out_channels
            # out_channels_mnet = 256
        elif pc_tokenizer_type == "merged-frozen":
            self.pc_tokenizer = FrozenPointCloudTokenizer()
            assert isinstance(self.config.merged_frozen_in_channels, list), "merged_frozen_in_channels must be a list of input frozen feature dimensions"
            self.pre_proj = nn.ModuleList([
                nn.Linear(d, self.config.hidden_size) for d in self.config.merged_frozen_in_channels
            ])
            out_channels_mnet = self.config.hidden_size
        else:
            raise ValueError(f"Unknown point cloud tokenizer type: {pc_tokenizer_type}")

        self.pc_tokenizer.load_pretrained()

        self.config.out_channels_mnet = out_channels_mnet

    def _init_adapter(self):
        adapter_type = self.config.adapter_type
        if adapter_type == "ffn":
            self.linear_3d = nn.Sequential(
                nn.LayerNorm(self.config.out_channels_mnet),
                nn.Linear(self.config.out_channels_mnet, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )
            self.linear_focus_bbox = nn.Sequential(
                nn.Linear(6, self.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )
        elif adapter_type == "linear":
            # self.linear_3d = nn.Linear(out_channels_mnet + 3, self.config.hidden_size)
            self.linear_3d = nn.Linear(self.config.out_channels_mnet, self.config.hidden_size)
            self.linear_focus_bbox = nn.Linear(6, self.config.hidden_size)
        elif adapter_type == "upsampler":
            # upsample_ratio = self.config.get("upsample_ratio", 2)
            upsample_ratio = getattr(self.config, "upsample_ratio", 2)
            self.linear_3d = nn.Sequential(
                # nn.LayerNorm(out_channels_mnet),
                nn.Linear(self.config.out_channels_mnet, self.config.hidden_size),
                nn.GELU(),
                # nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size * upsample_ratio),
                nn.GELU(),
            )
            self.linear_focus_bbox = nn.Sequential(
                nn.Linear(6, self.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
            )
            self.upsample_ratio = upsample_ratio
        elif adapter_type == "deformable-detr":
            raise NotImplementedError("Deformable DETR adapter is not implemented yet")
        elif adapter_type == "qformer" or adapter_type == "moe-qformer":
            # qformer
            # pretrained_qformer_name: Optional[str] = self.config.get("pretrained_qformer", None)
            pretrained_qformer_name: Optional[str] = getattr(self.config, "pretrained_qformer", None)
            qformer_config = InstructBlipQFormerConfig() # by default 12 layers, 768 hidden size == bert-base
            qformer_config.encoder_hidden_size = qformer_config.hidden_size
            # qformer_config.num_hidden_layers = self.config.get("qformer_num_hidden_layers", 12)
            qformer_config.num_hidden_layers = getattr(self.config, "qformer_num_hidden_layers", 12)

            if adapter_type == "moe-qformer":
                # build moe qformer
                self.qformer = build_moe_qformer_from_config(
                    pretrained_qformer_name, 
                    qformer_config, 
                    num_experts=4, 
                    num_experts_per_tok=2, 
                    init_with_same_pretrained_weights=getattr(self.config, "init_with_same_pretrained_weights", False)
                )
            else:
                if pretrained_qformer_name is not None:
                    logger.info(f"Loading pretrained qformer from {pretrained_qformer_name}")
                    self.qformer = InstructBlipQFormerModel.from_pretrained(pretrained_qformer_name, config=qformer_config)
                else:
                    logger.info(f"Building new qformer with config: {qformer_config}")
                    self.qformer = InstructBlipQFormerModel(qformer_config)
            self.qformer_query_tokens = nn.Parameter(torch.zeros(1, self.config.num_query_tokens, qformer_config.hidden_size))
            self.qformer_query_tokens.data.normal_(mean=0.0, std=0.02) # ? do we need to initialize the query tokens?
            self.qformer_to_language = nn.Linear(qformer_config.hidden_size, self.config.hidden_size)


            self.linear_3d = nn.Sequential(
                nn.Linear(self.config.out_channels_mnet, qformer_config.hidden_size),
                nn.GELU(),
                nn.Linear(qformer_config.hidden_size, qformer_config.hidden_size),
                nn.GELU(),
            )
            self.linear_focus_bbox = nn.Sequential(
                nn.Linear(6, qformer_config.hidden_size),
                nn.GELU(),
                nn.Linear(qformer_config.hidden_size, qformer_config.hidden_size),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
    def load_detector(self):
        if getattr(self, "pc_tokenizer", None) is not None:
            self.pc_tokenizer.load_pretrained()


class Fuyu3DCausalLMv2(FuyuPreTrainedModel, PCTokenizerAdapterMixin):
    # def __init__(self, config, **kwargs):
    def __init__(self, pretrained_args, **kwargs):
        super().__init__(FuyuConfig(**pretrained_args))
        # super().__init__(config)
        print(pretrained_args)
        # print kwargs without vocab
        print({k: v for k, v in kwargs.items() if k != "vocab"})
        self.fuyu: FuyuForCausalLM = FuyuForCausalLM.from_pretrained(**pretrained_args)
        
        # inherit from FuyuForCausalLM
        self.config = self.fuyu.config
        self.padding_idx = self.fuyu.padding_idx
        # self.vocab_size = self.fuyu.vocab_size
        self.gradient_checkpointing = self.fuyu.gradient_checkpointing

        self.config.iosa_threshold = kwargs.get("iosa_threshold", 0.25)

        self.config.mnet_path = kwargs.get("mnet_path", "/scratch/generalvision/mowentao/ScanQA/weights.pth")
        self.config.pnpp_path = kwargs.get("pnpp_path", "...")
        self.config.vote2cap_detr_path = kwargs.get("vote2cap_detr_path", "...")

        self.config.freeze_vote2cap_detr = kwargs.get("freeze_vote2cap_detr", True)
        self.config.freeze_mnet = kwargs.get("freeze_mnet", True)
        self.config.freeze_pnpp = kwargs.get("freeze_pnpp", True)

        self.config.num_query_tokens = kwargs.get("num_query_tokens", 128)
        self.config.upsample_ratio = kwargs.get("upsample_ratio", 2)
        self.config.use_focus_bbox = kwargs.get("use_focus_bbox", False)
        self.config.pretrained_qformer = kwargs.get("pretrained_qformer", None)
        self.config.qformer_num_hidden_layers = kwargs.get("qformer_num_hidden_layers", 12)

        # 3D scene encoding
        self.config.in_channels = kwargs.get("in_channels", 3)
        self.config.spatial_patch_size = kwargs.get("spatial_patch_size", 24)
        self.config.pooling_method = kwargs.get("pooling_method", "max")
        self.config.vote2cap_return_type = kwargs.get("vote2cap_return_type", "enc_features")
        # self.config.out_channels = kwargs.get("out_channels", PointNetConfig().proposal_size)
        self.config.frozen_in_channels = kwargs.get("frozen_in_channels", 256)
        self.config.merged_frozen_in_channels = kwargs.get("merged_frozen_in_channels", [256, 256])
        pc_tokenizer_type = kwargs.get("pc_tokenizer_type", "minkowski")
        self.config.pc_tokenizer_type = pc_tokenizer_type

        self.config.keep_all_objects = kwargs.get("keep_all_objects", False)

        
        self._init_pc_tokenizer()
        # out_channels_mnet = self.config.out_channels_mnet

        self.config.adapter_type = kwargs.get("adapter_type", "ffn")
        self.adater_type = self.config.adapter_type
        

        self._init_adapter()

        
        self.use_focus_bbox = kwargs.get("use_focus_bbox", False)
        
        # TODO: shall we add 3D-patch xyz position embedding?

        self.reinforce = kwargs.get("reinforce", False)
        self.reinforce_sigma = kwargs.get("reinforce_sigma", 0.5)
        self.vocab = kwargs.get("vocab", None)
        if self.reinforce:
            assert self.vocab is not None, "vocab must be provided for REINFORCE training"
            print(f"REINFORCE training is enabled, {self.reinforce_sigma=}")

        self.build_think_token(kwargs.get("num_think_tokens", 0), self.config.hidden_size)

        self.config.use_2d = kwargs.get("use_2d", True)
        self.config.use_3d = kwargs.get("use_3d", True)

        self.config.coeff_frame_params = kwargs.get("coeff_frame_params", 0.1)
        
        ## 2D & 3D MASK SETTINGs
        self.config.p_drop_2d = kwargs.get("p_drop_2d", 0.0) # prob to drop ALL 2D tokens
        self.config.p_drop_3d = kwargs.get("p_drop_3d", 0.0) # prob to drop ALL 3D tokens

        self.config.do_drop_2d_partial = kwargs.get("do_drop_2d_partial", False) # do drop PARTIAL 2D tokens
        # if we do, the drop ratio ~ Beta(p_drop_2d_partial_alpha, p_drop_2d_partial_beta), mean = alpha / (alpha + beta)
        self.config.p_drop_2d_partial_alpha = kwargs.get("p_drop_2d_partial_alpha", 2.0) # prob to drop PARTIAL 2D tokens
        self.config.p_drop_2d_partial_beta = kwargs.get("p_drop_2d_partial_beta", 8.0) # prob to drop PARTIAL 2D tokens

        self.related_object_embedding = nn.Parameter(torch.zeros(1, self.config.hidden_size))
        nn.init.kaiming_normal_(self.related_object_embedding)
        self.related_object_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.out_channels_mnet),
        )

        self.config.choose_related_object = kwargs.get("choose_related_object", False)
        self.config.trim_objects = kwargs.get("trim_objects", True)

        if (to_remove_token_ids := kwargs.get("to_remove_token_ids", None)) is not None:
            if len(to_remove_token_ids) > 0:
                # remove token indices from embedding and lm_head
                to_remove_token_ids = sorted(to_remove_token_ids)
                logger.warning(f"Removing {len(to_remove_token_ids)} token indices: {to_remove_token_ids}")
                
                self._shrink_token_embeddings(to_remove_token_ids) # TODO: do until moved to GPU, much much faster

        self.text_tokenizer = kwargs.get("text_tokenizer", None)

        self.config.use_object_index_embedding = kwargs.get("use_object_index_embedding", False)
        if self.config.use_object_index_embedding:
            assert self.config.trim_objects, "we assume objects are no-masked starting from 0 to some N"
            self.object_index_embedding = nn.Embedding(512, self.config.hidden_size)
            self.fuyu.language_model.resize_token_embeddings(len(self.text_tokenizer))

        self.config.use_object_textual_index = kwargs.get("use_object_textual_index", False)
        if self.config.use_object_textual_index:
            # from tokenizer, find the digit token ids
            assert self.text_tokenizer is not None, "text_tokenizer must be provided for use_object_textual_index"
            # digit_ids = {
            #     i: self.text_tokenizer.convert_tokens_to_ids(str(i)) for i in range(256)
            # }

            logger.info(f"Adding new tokens embeddings for object textual index")
            # self.object_tokens = [f"<OBJ{i}>" for i in range(512)]
            # self.text_tokenizer.add_tokens(self.object_tokens)
            self.fuyu.language_model.resize_token_embeddings(len(self.text_tokenizer))
            self.object_index_embedding_start = self.text_tokenizer.convert_tokens_to_ids("<OBJ0>")
            # self.object_index_embedding_end = self.text_tokenizer.convert_tokens_to_ids("<OBJ511>")
            self.added_object_tokens = kwargs.get("added_object_tokens", 512)
            self.object_index_embedding_end = self.object_index_embedding_start + self.added_object_tokens - 1

            self.object_index_textual_embedding = nn.Embedding(self.added_object_tokens, self.config.hidden_size)
            
            # self.vocab_size = self.text_tokenizer.vocab_size

        self.config.use_grounding_classifier = kwargs.get("use_grounding_classifier", False)
        self.config.coeff_grounding_classifier = kwargs.get("coeff_grounding_classifier", 0.5)
        if self.config.use_grounding_classifier:
            # lowrank_dim = self.config.hidden_size // 32
            lowrank_dim = 64
            # self.grounding_lowrank_A = nn.Linear(self.config.hidden_size, lowrank_dim)
            self.grounding_lowrank_A = nn.Sequential(
                nn.Linear(self.config.hidden_size, lowrank_dim),
                nn.GELU(),
                nn.LayerNorm(lowrank_dim),
            )
            # self.grounding_lowrank_B = nn.Linear(self.config.hidden_size, lowrank_dim)
            self.grounding_lowrank_B = nn.Sequential(
                nn.Linear(self.config.hidden_size, lowrank_dim),
                nn.GELU(),
                nn.LayerNorm(lowrank_dim),
            )
            self.grounding_bilinear = nn.Bilinear(lowrank_dim, lowrank_dim, lowrank_dim)
            self.grounding_up_proj = nn.Linear(lowrank_dim, self.config.hidden_size)
            # self.grounding_head = nn.Linear(lowrank_dim, 2)

            self.grounding_head = nn.Linear(self.config.hidden_size, 2)
            

        # self.vocab_size = self.text_tokenizer.vocab_size if self.config.use_object_textual_index else self.config.vocab_size

    def fuse_for_grounding(self, object_embeds, text_embeds):
        # object_embeds: [B, N, H]
        # text_embeds: [B, H]
        # return: [B, N, H]
        # complexity: O(B * N * d_lowrank ** 3)
        x_A = self.grounding_lowrank_A(object_embeds) # [B, N, H] -> [B, N, d_lowrank]
        x_B = self.grounding_lowrank_B(text_embeds) # [B, H] -> [B, d_lowrank]
        # x_A = F.layer_norm(F.gelu(x_A)) # [B, N, d_lowrank]
        # x_B = F.layer_norm(F.gelu(x_B)) # [B, d_lowrank] 
        x = self.grounding_bilinear(x_A, x_B.unsqueeze(1).expand(-1, x_A.size(1), -1)) # [B, N, d_lowrank]
        x = self.grounding_up_proj(F.gelu(x)) + object_embeds # [B, N, H]
        x = self.grounding_head(x)
        return x

    def _get_shrinked_token_embeddings(self, old_embeddings: nn.Embedding, to_remove_token_ids):
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        new_num_tokens = old_num_tokens - len(to_remove_token_ids)
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embedding_dim,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # copy token embeddings from the previous weights
        old_indices = [idx for idx in range(old_num_tokens) if idx not in to_remove_token_ids]
        new_indices = [idx for idx in range(new_num_tokens)]
        new_embeddings.weight.data[new_indices, :] = old_embeddings.weight.data[old_indices, :]

        old_embeddings.weight.data = new_embeddings.weight.data
        old_embeddings.num_embeddings = new_embeddings.weight.data.shape[0]
        # if old_embeddings.padding_idx is not None and (new_num_tokens - 1) < old_embeddings.padding_idx:
        #     old_embeddings.padding_idx = None

        # update padding_idx, by reduce its value by the number of tokens removed before it
        if old_embeddings.padding_idx is not None:
            old_embeddings.padding_idx -= sum(1 for idx in to_remove_token_ids if idx < old_embeddings.padding_idx)
            if old_embeddings.padding_idx < 0: # if padding_idx is removed
                old_embeddings.padding_idx = None

        return old_embeddings
    
    def _get_shrinked_lm_head(self, old_lm_head: nn.Linear, to_remove_token_ids, transposed=False):
        old_num_tokens, old_lm_head_dim = (
            old_lm_head.weight.size() if not transposed else old_lm_head.weight.t().size()
        )
        new_num_tokens = old_num_tokens - len(to_remove_token_ids)

        # Build new lm head
        new_lm_head_shape = (old_lm_head_dim, new_num_tokens) if not transposed else (new_num_tokens, old_lm_head_dim)
        has_new_lm_head_bias = old_lm_head.bias is not None

        new_lm_head = nn.Linear(
            *new_lm_head_shape,
            bias=has_new_lm_head_bias,
            device=old_lm_head.weight.device,
            dtype=old_lm_head.weight.dtype,
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_lm_head)

        # copy token embeddings from the previous weights
        old_indices = [idx for idx in range(old_num_tokens) if idx not in to_remove_token_ids]
        new_indices = [idx for idx in range(new_num_tokens)]
        new_lm_head.weight.data[new_indices, :] = old_lm_head.weight.data[old_indices, :]

        if has_new_lm_head_bias:
            new_lm_head.bias.data[new_indices] = old_lm_head.bias.data[old_indices]

        return new_lm_head

    def _shrink_token_embeddings(self, to_remove_token_ids):
        logger.info(f"Shrinking token embeddings")
        old_embeddings = self.fuyu.language_model.get_input_embeddings()
        new_embeddings = self._get_shrinked_token_embeddings(old_embeddings, to_remove_token_ids)

        old_embeddings_requires_grad = old_embeddings.weight.requires_grad
        new_embeddings.requires_grad_(old_embeddings_requires_grad)
        self.fuyu.language_model.set_input_embeddings(new_embeddings)

        # if word embeddings are not tied, make sure that lm head is resized as well
        if self.fuyu.language_model.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            logger.info("Resizing LM head to match new embeddings")
            old_lm_head = self.fuyu.language_model.get_output_embeddings()
            new_lm_head = self._get_shrinked_lm_head(old_lm_head, to_remove_token_ids)

            old_lm_head_requires_grad = old_lm_head.weight.requires_grad
            new_lm_head.requires_grad_(old_lm_head_requires_grad)
            self.fuyu.language_model.set_output_embeddings(new_lm_head)

        return self.fuyu.language_model.get_input_embeddings()

    @property
    def vocab_size(self):
        return self.fuyu.language_model.config.vocab_size

    def to_bfloat16(self):
        """
        cast non-LVLM parameters to bfloat16
        """
        self.linear_3d = self.linear_3d.to(dtype=torch.bfloat16)
        if getattr(self, "pc_tokenizer", None) is not None:
            self.pc_tokenizer.to(dtype=torch.bfloat16)
        if self.use_focus_bbox:
            self.linear_focus_bbox = self.linear_focus_bbox.to(dtype=torch.bfloat16)

        if self.think_tokens is not None:
            self.think_tokens = self.think_tokens.to(dtype=torch.bfloat16)


    def build_think_token(self, num_think_tokens, output_hidden_size):
        self.num_think_tokens = num_think_tokens
        self.think_tokens = None
        # think tokens
        if num_think_tokens > 0:
            eos_tokens = torch.nn.Parameter(torch.randn(1, num_think_tokens, output_hidden_size))
            nn.init.trunc_normal_(eos_tokens, mean=0.0, std=0.02)
            self.think_tokens = eos_tokens

    def embed_tokens(self, input_ids):
        if self.config.use_object_textual_index and is_object_index.any():
            is_object_index = input_ids >= self.object_index_embedding_start and input_ids <= self.object_index_embedding_end
            if is_object_index.any():
                pass
        
        #TODO: how to separate added token and original token?
            
        
        return self.fuyu.language_model.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        focus_bbox: Optional[torch.FloatTensor] = None, # [B, 6]
        focus_bbox_mask: Optional[torch.BoolTensor] = None, # [B]
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mnet_inputs: Optional[Tuple] = None, # tuple of (coords, feats)
        qformer_inputs: Optional[dict] = None, # dict of inputs for qformer from tokenization
        frame_caption_mask: Optional[torch.BoolTensor] = None, # [B], mask for frame caption - True to mask off non-view objects
        frame_intrinsics: Optional[torch.FloatTensor] = None, # [B, 4, 4] camera intrinsics
        frame_poses: Optional[torch.FloatTensor] = None, # [B, 4, 4] camera pose
        axis_alignments: Optional[torch.FloatTensor] = None, # [B, 4, 4] axis alignments
        additional_attention_weights: Optional[torch.FloatTensor] = None, # [B, seq_len, seq_len], additional attention weights
        related_object_bboxes: Optional[List[torch.FloatTensor]] = None, # [B, <variable N>, 6], related object bboxes for QA
        target_object_indices: Optional[torch.LongTensor] = None, # [B], target object indices for instruction. corresponds to the <OBJ{i}> token
        **kwargs, # hack to reduce the need to manually change the code 
    ) -> Union[Tuple, BaseModelOutputWithPast, dict]:
        
        is_in_beam_search = False

        if mnet_inputs is not None:
            if self.config.pc_tokenizer_type != "frozen":
                pc_tokenizer_output = self.pc_tokenizer(mnet_inputs)
            else:
                pc_tokenizer_output = mnet_inputs

            if len(pc_tokenizer_output) == 2:
                out, coords_mask = pc_tokenizer_output
                predicted_bbox_corners = None
            elif len(pc_tokenizer_output) == 3:
                out, coords_mask, predicted_bbox_corners = pc_tokenizer_output

            if self.config.pc_tokenizer_type == "frozen" and self.config.trim_objects:
                # trim objects to shrink length
                out, coords_mask, predicted_bbox_corners = trim_objects(out, coords_mask, predicted_bbox_corners)

            if self.config.keep_all_objects:
                coords_mask = torch.ones_like(coords_mask)

            if self.config.pc_tokenizer_type == "merged-frozen":
                out = torch.cat([proj(x) for proj, x in zip(self.pre_proj, out)], dim=1) # [B, N1, H] + [B, N2, H] + ... -> [B, N, H]
                coords_mask = torch.cat([mask for mask in coords_mask], dim=1)
                if predicted_bbox_corners is not None:
                    predicted_bbox_corners = torch.cat([bbox_corners for bbox_corners in predicted_bbox_corners], dim=1)

            # print_once(f"predicted_bbox_corners: {predicted_bbox_corners.shape if predicted_bbox_corners is not None else None}")
            # filter objects in the view
            if predicted_bbox_corners is not None and frame_caption_mask is not None:
                # check if frame_caption_mask has any True value
                if frame_caption_mask.any():
                    view_object_mask, projected_bbox = calculate_in_view_objects(
                        predicted_bbox_corners, 
                        frame_intrinsics, 
                        frame_poses, 
                        axis_alignments, 
                        iosa_threshold=self.config.iosa_threshold
                    ) # [B, N]
                    view_object_mask[~frame_caption_mask] = True # for non-frame caption samples, always keep all objects
                    # mask off the objects that are not in the view
                    coords_mask = coords_mask & view_object_mask

            # add special embedding if box is related to task
            if predicted_bbox_corners is not None and related_object_bboxes is not None:
                relative_object_mask = calculate_related_objects(
                    predicted_bbox_corners, 
                    related_object_bboxes, 
                    iou_threshold=0.25, 
                ) # [B, N], 0 or (min_iou-1) for related objects
                if not self.config.choose_related_object:
                    added_value = self.related_object_embedding.expand(out.shape[0], -1, -1) # [B, 1, H]
                    added_value = self.related_object_mlp(added_value * relative_object_mask.unsqueeze(2))
                    out = out + added_value 
                else:
                    out, coords_mask = choose_related_objects(out, coords_mask, relative_object_mask)

            # logger.info(coords_mask.float().mean().item())

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                # NOTE: in Minkowski case, the input SparseTensor can't be automatically expanded
                #       so we need to repeat the coords_mask and out manually
                #       in PointNet++ case, the input is auto expanded, so the code will not reach here
                beam_size = input_ids.shape[0] // coords_mask.shape[0]
                print_once(f"Beam search detected, {beam_size=}")

            
            linear_3d_weight_dtype = self.linear_3d[0].weight.dtype if isinstance(self.linear_3d, nn.Sequential) else self.linear_3d.weight.dtype
            if out.dtype != linear_3d_weight_dtype:
                out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage
            
            out = self.linear_3d(out) # of config.hidden_size dim, [B, N, H]
            # scene_embed_indices = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N]
            scene_embed_mask = coords_mask.clone().detach()
            # grounding_label = torch.zeros_like(coords_mask, dtype=out.dtype).detach()
            # if target_object_indices is not None:
            #     for i in range(out.shape[0]):
            #         grounding_label[target_object_indices[i]] = 1.0

            if is_in_beam_search:
                out = out.repeat_interleave(beam_size, dim=0)
                coords_mask = coords_mask.repeat_interleave(beam_size, dim=0)
                # TODO: also repeat grounding label?

            # add object index embedding
            if self.config.use_object_index_embedding:
                # object_indices = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(out.shape[0], -1)
                # object_indices = self.object_index_embedding(object_indices) # [B, N, H]

                # DEBUG: cheat to add target_object_indices
                if (debug := False) and target_object_indices is not None:
                    target_object_indices = target_object_indices.unsqueeze(1) # [B, 1]
                    target_object_indices = self.object_index_embedding(target_object_indices) # [B, 1, H]
                    object_indices = target_object_indices
                else:
                    object_indices = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N]
                    object_indices = self.object_index_embedding(object_indices) # [B, N, H]

                out = out + object_indices

            if self.config.use_object_textual_index:
                # object_indices = [str(i) for i in range(out.shape[1])]
                object_indices = [f"<OBJ{i}>" for i in range(out.shape[1])]
                object_indices = [self.text_tokenizer.encode(obj, add_special_tokens=False) for obj in object_indices]
                # we assume all object index string is tokenized into same length of tokens
                # object_indices = self.text_tokenizer.convert_tokens_to_ids(object_indices)
                len_obj_tokens = len(object_indices[0])
                print_once(f"object token length: {len_obj_tokens}") # if added tokens, shall be 1
                if len_obj_tokens > 1:
                    logger.warning(f"object token length is {len_obj_tokens}, shall be 1, check if there are more object tokens than added token indices")

                all_object_indices = torch.tensor(sum(object_indices, []), device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N_objects * len_obj_tokens]
                # object_indices = torch.tensor(object_indices, device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N_objects]
                object_index_embeds = self.fuyu.language_model.get_input_embeddings()(all_object_indices).to(out) # [B, N_objects * len_obj_tokens, H]
                # added_object_embeds = self.fuyu.language_model.get_input_embeddings().weight[self.object_index_embedding_start:self.object_index_embedding_end+1]


                coords_mask = coords_mask.repeat_interleave((len_obj_tokens + 1), dim=1) # [B, N_objects] -> [B, N_objects * (len_obj_tokens + 1)]
                # ic(coords_mask.float().mean().item())

                # Interleave object indices with out
                interleaved = torch.zeros((out.shape[0], out.shape[1] * (len_obj_tokens + 1), out.shape[2]), device=out.device, dtype=out.dtype) # [B, N_objects * (len_obj_tokens + 1), H]
                # interleaved[:, 0::2] = object_index_embeds
                # interleaved[:, 1::2] = out
                # interleaved[:, 0::(len_obj_tokens + 1)] = object_index_embeds
                # interleaved[:, len_obj_tokens::(len_obj_tokens + 1)] = out
                # scene_embed_mask = torch.zeros(out.shape[0], out.shape[1] * (len_obj_tokens + 1), out.shape[2], dtype=torch.bool, device=out.device)
                # scene_embed_mask[:, len_obj_tokens::(len_obj_tokens + 1)] = True # place to put scene embeddings
                
                # ic(out.shape, object_index_embeds.shape, scene_embed_mask.shape, interleaved.shape)

                # interleaved[scene_embed_mask] = out
                # interleaved[~scene_embed_mask] = object_index_embeds
                interleaved[:, 0::(len_obj_tokens + 1)] = object_index_embeds
                interleaved[:, 1::(len_obj_tokens + 1)] = out

                # Update out with the interleaved features
                out = interleaved
                scene_embed_mask = torch.repeat_interleave(scene_embed_mask, len_obj_tokens + 1, dim=1)
                # grounding_label = torch.repeat_interleave(grounding_label, len_obj_tokens + 1, dim=1)
                if target_object_indices is not None:
                    target_object_indices = target_object_indices * (len_obj_tokens + 1) + 1 # +1 to skip the object index token


            if self.config.adapter_type == "upsampler":
                # [B, L, H*upsample_ratio] -> [B, L*upsample_ratio, H]
                out = out.view(out.shape[0], -1, self.config.hidden_size) # [B, L * upsample_ratio, H]
                # 0, ..., upsample_ratio-1 ~ 0-th token, mask=coords_mask[..., 0]
                # upsample_ratio, ..., 2*upsample_ratio-1 ~ 1-th token, mask=coords_mask[..., 1]
                # thus repeat coords_mask with upsample_ratio times

                coords_mask = coords_mask.repeat_interleave(self.upsample_ratio, dim=1)
                scene_embed_mask = scene_embed_mask.repeat_interleave(self.upsample_ratio, dim=1)
                # grounding_label = grounding_label.repeat_interleave(self.upsample_ratio, dim=1)
                if target_object_indices is not None:
                    target_object_indices = target_object_indices * self.upsample_ratio
            
            
            if self.use_focus_bbox:
                if focus_bbox is not None:
                    focus_bbox = focus_bbox.to(out.dtype).unsqueeze(1)
                    # maskoff the focus_bbox
                    if focus_bbox_mask is not None:
                        coords_mask = torch.cat((coords_mask, focus_bbox_mask.unsqueeze(1).to(coords_mask)), dim=1)
                    else:
                        coords_mask = torch.cat((coords_mask, torch.ones(coords_mask.shape[0], 1, dtype=torch.bool, device=coords_mask.device)), dim=1)
                else:
                    focus_bbox = torch.zeros(out.shape[0], 1, 6, dtype=out.dtype, device=out.device)
                    coords_mask = torch.cat((coords_mask, torch.zeros(coords_mask.shape[0], 1, dtype=torch.bool, device=coords_mask.device)), dim=1)

                focus_bbox = self.linear_focus_bbox(focus_bbox)
                out = torch.cat((out, focus_bbox), dim=1)
                scene_embed_mask = torch.cat((scene_embed_mask, torch.zeros(scene_embed_mask.shape[0], 1, dtype=torch.bool, device=scene_embed_mask.device)), dim=1)
                # grounding_label = torch.cat((grounding_label, torch.zeros(grounding_label.shape[0], 1, dtype=grounding_label.dtype, device=grounding_label.device)), dim=1)

            if self.config.adapter_type == "qformer" or self.config.adapter_type == "moe-qformer":
                # NOTE: DO WE NEED REPEAT_INTERLEAVE HERE?
                if is_in_beam_search:
                    # repeat input ids and attention mask
                    qformer_inputs["input_ids"] = qformer_inputs["input_ids"].repeat_interleave(beam_size, dim=0)
                    qformer_inputs["attention_mask"] = qformer_inputs["attention_mask"].repeat_interleave(beam_size, dim=0)
                    # print(out.shape, qformer_inputs["input_ids"].shape, qformer_inputs["attention_mask"].shape)

                # prepare query_attention_mask
                query_attention_mask = torch.ones(out.shape[0], self.config.num_query_tokens, dtype=torch.bool, device=out.device)
                query_attention_mask = torch.cat((query_attention_mask, qformer_inputs["attention_mask"]), dim=1)

                out = self.qformer(
                    # input_ids=None, # TODO: add input_ids
                    input_ids=qformer_inputs["input_ids"],
                    attention_mask=query_attention_mask,
                    query_embeds=self.qformer_query_tokens.expand(out.shape[0], -1, -1),
                    encoder_hidden_states=out,
                    encoder_attention_mask=coords_mask,
                    return_dict=True,
                ).last_hidden_state
                out = self.qformer_to_language(out)

                # replace coords_mask as all True
                coords_mask = torch.ones(*out.shape[:2], dtype=torch.bool, device=out.device)
            

            if self.num_think_tokens > 0:
                think_tokens = self.think_tokens.expand(out.shape[0], -1, -1)
                out = torch.cat((out, think_tokens), dim=1)
                coords_mask = torch.cat((coords_mask, torch.ones(coords_mask.shape[0], self.num_think_tokens, dtype=torch.bool, device=coords_mask.device)), dim=1)
                scene_embed_mask = torch.cat((scene_embed_mask, torch.zeros(scene_embed_mask.shape[0], self.num_think_tokens, dtype=torch.bool, device=scene_embed_mask.device)), dim=1)
                # len_scene_embeddings += self.num_think_tokens
            
            len_scene_embeddings = out.shape[1]
            self.cached_coords_mask = coords_mask.detach().cpu().clone().numpy() # if in beam search, this is repeated
            self.cached_len_scene_embeddings = len_scene_embeddings
        else:
            coords_mask: np.ndarray = self.cached_coords_mask.copy() # use cached coords mask from previous generation step
            coords_mask = torch.from_numpy(coords_mask).to(input_ids.device)
            scene_embed_mask = None
            len_scene_embeddings = self.cached_len_scene_embeddings

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                beam_size = input_ids.shape[0] // coords_mask.shape[0]
                print_once(f"Beam search detected in later tokens, {beam_size=}")

            out = None
            

        # Fuyu LVLM
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True # always output hidden states for now
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # seq_length += len_scene_embeddings # to generate appropriate position_ids including scene embeddings

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None: # training
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

            position_ids = self.insert_position_ids(
                image_patch_input_indices=image_patches_indices, 
                position_ids=position_ids, 
                len_scene_embeddings=len_scene_embeddings,
                scene_embedding_mask=coords_mask,
            )

        else: # generation
            # add len_scene_embeddings to position_ids to generate appropriate position_ids including scene embeddings
            # position_ids = position_ids + len_scene_embeddings 
            # NOTE: only execute when past_key_values is None, otherwise position_ids is already [1] length
            if past_key_values is None: # 1st step, position_ids 
                position_ids = self.insert_position_ids(
                    image_patch_input_indices=image_patches_indices, 
                    position_ids=position_ids, 
                    len_scene_embeddings=len_scene_embeddings,
                    scene_embedding_mask=coords_mask,
                )
            else: # later steps, position_ids is already [1] length
                num_scene_embeddings = coords_mask.sum(dim=-1).unsqueeze(1)
                position_ids = position_ids + num_scene_embeddings

        
        # insert scene attention_mask, this is a must
        attention_mask, scene_embed_mask = self.insert_attention_mask(
            image_patch_input_indices=image_patches_indices, 
            attention_mask=attention_mask,
            scene_attention_mask=coords_mask,
            scene_embed_mask=scene_embed_mask,
            use_3d=self.config.use_3d,
            use_2d=self.config.use_2d,
            p_drop_2d=self.config.p_drop_2d if self.training else 0.0,
            p_drop_3d=self.config.p_drop_3d if self.training else 0.0,
            do_drop_2d_partial=self.config.do_drop_2d_partial if self.training else False,
            p_drop_2d_partial_alpha=self.config.p_drop_2d_partial_alpha,
            p_drop_2d_partial_beta=self.config.p_drop_2d_partial_beta,
        )


        if inputs_embeds is None: # training or generation 1st step
            inputs_embeds = self.fuyu.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                # print(len(image_patches))
                patch_embeddings = [
                    self.fuyu.vision_embed_tokens(patch.to(self.fuyu.vision_embed_tokens.weight.dtype)).squeeze(0)
                    for patch in image_patches
                ]
                # if is_in_beam_search:
                is_in_image_beam_search = input_ids.shape[0] != len(patch_embeddings)
                if is_in_image_beam_search:
                    print_once(f"Beam search detected in image patches, {input_ids.shape[0]=}, {len(patch_embeddings)=}")
                    image_beam_size = input_ids.shape[0] // len(patch_embeddings)
                    # repeat_interleave for beam search
                    patch_embeddings = [patch_embeddings[i // image_beam_size] for i in range(len(patch_embeddings) * image_beam_size)]
                
                # replace word embeddings with image patch embeddings
                #   and insert scene embeddings & labels
                inputs_embeds, _, labels = self.gather_continuous_embeddings(
                    word_embeddings=inputs_embeds,
                    continuous_embeddings=patch_embeddings,
                    image_patch_input_indices=image_patches_indices,
                    scene_embeddings=out,
                    scene_attention_mask=None,
                    attention_mask=None,
                    labels=labels,
                    position_ids=None,
                )
        
        # ic(inputs_embeds.shape, attention_mask.shape, position_ids.shape)

        outputs = self.fuyu.language_model(
            inputs_embeds=inputs_embeds,
            labels=None, # labels if (self.training and not self.reinforce) else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            # additional_attention_weights=additional_attention_weights,
        )
        outputs = outputs.__dict__
        outputs = MyObjectDict(outputs)
        # print(self.training)


        if self.training:
            logits = outputs.logits if return_dict else outputs[0]
            _label_shape = labels.shape
            if self.reinforce:
                # calculate reinforce loss
                # print(labels.shape)
                reward_labels = batch_calculate_reinforce_reward_labels(labels.view(-1), self.vocab, sigma=self.reinforce_sigma)
                reward_labels = reward_labels.view(*_label_shape, -1).to(logits) # [B, N, V]
                # loss = loss_reinforce(logits, reward_labels)
                # shift one for auto-regressive
                shift_logits = logits[..., :-1, :].contiguous()
                shift_reward_labels = reward_labels[..., 1:, :].contiguous()

                loss_fct = nn.CrossEntropyLoss(reduction="sum")
                # loss = loss_fct(logits.view(-1, logits.size(-1)), reward_labels.view(-1, reward_labels.size(-1))) # REINFORCE loss
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_reward_labels.view(-1, shift_reward_labels.size(-1))) # REINFORCE loss
                # mean by non-mask tokens
                unmasked_tokens = (labels != -100).sum()
                # print(unmasked_tokens)
                loss = loss / unmasked_tokens
            else:
                # if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels) # [B * N]

                if (calc_train_token_acc := True) and self.training:
                    # calculate token accuracy
                    pred = shift_logits.argmax(dim=-1)
                    correct = (pred == shift_labels) & (shift_labels != -100) # mask off -100
                    correct = correct.sum().item()
                    total = (shift_labels != -100).sum().item()
                    token_acc = correct / total
                    # ic(correct, total, token_acc)
                    average_meter.update('correct', correct, type="integer")
                    average_meter.update('total', total, type="integer")
                    average_meter.update('token_acc', token_acc, type="percent")

        if self.config.use_grounding_classifier and scene_embed_mask is not None:
            # NOTE: after 1st step generation, scene_embed_mask is already inserted and None
            assert target_object_indices is not None, "target_object_indices must be provided for grounding classifier"
            last_hidden_state = outputs.hidden_states[-1] # [B, L, H]
            # fuse with last instruction's token embeddings
            if labels is not None:
                label_start_tokens = []
                for i in range(last_hidden_state.shape[0]):
                    label_start_token = (labels[i] != -100).nonzero(as_tuple=True)[0][0] # first token with label, the start of response
                    label_start_tokens.append(label_start_token - 1) # -1 to get the last token of the instruction

                label_start_tokens = torch.tensor(label_start_tokens, device=last_hidden_state.device)
                
            else:
                # last instruction token?
                # label_start_tokens = torch.full((last_hidden_state.shape[0],), last_hidden_state.shape[1] - 1, dtype=torch.long, device=last_hidden_state.device)
                # last token with non-zero attention

                label_start_tokens = []
                for i in range(last_hidden_state.shape[0]):
                    label_start_token = (attention_mask[i] != 0).nonzero(as_tuple=True)[0][-1]
                    label_start_tokens.append(label_start_token)

                label_start_tokens = torch.tensor(label_start_tokens, device=last_hidden_state.device)
                # ic(label_start_tokens, last_hidden_state.shape)

            label_start_tokens_embeds = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), label_start_tokens] # [B, H]

            # grounding_logits = self.grounding_head(last_hidden_state) # [B, L, 2]
            grounding_logits = self.fuse_for_grounding(last_hidden_state, label_start_tokens_embeds) # [B, L, 2]

            # calculate grounding loss
            grounding_labels = torch.full(grounding_logits.shape[:2], -100, dtype=grounding_logits.dtype, device=grounding_logits.device)
            scene_embed_start_idxs = []
            for i in range(grounding_labels.shape[0]):
                # find scene embeddings start position
                scene_embed_start_idx = scene_embed_mask[i].nonzero(as_tuple=True)[0][0]
                scene_embed_end_idx = scene_embed_mask[i].nonzero(as_tuple=True)[0][-1]
                scene_embed_start_idxs.append(scene_embed_start_idx)

                grounding_labels[i, scene_embed_start_idx: scene_embed_end_idx + 1] = 0 # not the correct object
                grounding_labels[i, scene_embed_start_idx + target_object_indices[i]] = 1 # correct object

            grounding_logits[grounding_labels == -100] = -1e9
            scene_embed_start_idxs = torch.tensor(scene_embed_start_idxs, device=last_hidden_state.device)

            # softmax grounding_logits
            grounding_softmax_label = scene_embed_start_idxs + target_object_indices # [B]
            # ic(scene_embed_start_idxs, target_object_indices, grounding_softmax_label)
            # ic(grounding_logits[..., 0].argmax(dim=-1), grounding_softmax_label)
            grounding_softmax_loss_fct = nn.CrossEntropyLoss()
            grounding_softmax_loss = grounding_softmax_loss_fct(grounding_logits[..., 0], grounding_softmax_label)

            grounding_softmax_correct = (grounding_logits[..., 0].argmax(dim=-1) == grounding_softmax_label).sum().item()
            grounding_softmax_total = grounding_softmax_label.shape[0]
            grounding_acc_softmax = grounding_softmax_correct / grounding_softmax_total

            # ic(scene_embed_start_idxs, target_object_indices)

            from torchvision.ops import sigmoid_focal_loss
            grounding_labels = grounding_labels.view(-1)
            grounding_logits = grounding_logits.view(-1, 2)[..., 0]

            grounding_labels_non_ignore = grounding_labels[grounding_labels != -100]
            grounding_logits_non_ignore = grounding_logits[grounding_labels != -100]

            alpha = 0.99
            grounding_loss = sigmoid_focal_loss(grounding_logits_non_ignore, grounding_labels_non_ignore, reduction="mean", gamma=4.0, alpha=alpha)
            grounding_loss = grounding_loss / (1-alpha)
            
            # grounding_loss_fct = nn.CrossEntropyLoss()
            # grounding_loss = grounding_loss_fct(grounding_logits.view(-1, 2), grounding_labels.view(-1).long())
            if self.training:
                # loss = loss + self.config.coeff_grounding_classifier * (grounding_loss + grounding_softmax_loss)
                loss = loss + self.config.coeff_grounding_classifier * grounding_softmax_loss

            # outputs["loss_grounding"] = grounding_loss

            # TODO: calculate grounding output: which is of highest value
            # grounding_output = torch.max(grounding_logits, dim=-1).indices

            # calculate pos/neg
            # grounding_pred = grounding_logits.argmax(dim=-1)
            grounding_pred = (grounding_logits_non_ignore > 0.0).long()
            grounding_correct = (grounding_pred == grounding_labels_non_ignore)
            grounding_correct = grounding_correct.sum().item()
            grounding_total = (grounding_labels_non_ignore != -100).sum().item()

            # grounding_correct = (grounding_pred == grounding_labels) & (grounding_labels != -100) # mask off -100
            # grounding_correct = grounding_correct.sum().item()
            # grounding_total = (grounding_labels != -100).sum().item()
            grounding_acc = grounding_correct / grounding_total

            # ic(grounding_correct, grounding_total, grounding_acc)

            # grounding_correct_postive = (grounding_pred == 1) & (grounding_labels == 1)
            # grounding_correct_postive = grounding_correct_postive.sum().item()
            # grounding_total_postive = (grounding_labels == 1).sum().item()

            grounding_correct_positive = (grounding_pred == 1) & (grounding_labels_non_ignore == 1)
            grounding_correct_positive = grounding_correct_positive.sum().item()
            grounding_total_positive = (grounding_labels_non_ignore == 1).sum().item()
            grounding_acc_positive = grounding_correct_positive / grounding_total_positive

            # ic(grounding_correct_postive, grounding_total_postive, grounding_acc_positive)

            grounding_correct_negative = (grounding_pred == 0) & (grounding_labels_non_ignore == 0)
            grounding_correct_negative = grounding_correct_negative.sum().item()
            # grounding_total_negative = (grounding_labels == 0).sum().item()
            grounding_total_negative = (grounding_labels_non_ignore == 0).sum().item()
            grounding_acc_negative = grounding_correct_negative / grounding_total_negative

            # ic(grounding_correct_negative, grounding_total_negative, grounding_acc_negative)

            if debug:= True or (not self.training):
                # report grounding accuracies
                # ic(grounding_acc_softmax, grounding_acc, grounding_acc_positive, grounding_acc_negative)
                average_meter.update('grounding_acc_softmax', grounding_acc_softmax, type="percent")
                average_meter.update('grounding_acc', grounding_acc, type="percent")
                average_meter.update('grounding_acc_positive', grounding_acc_positive, type="percent")
                average_meter.update('grounding_acc_negative', grounding_acc_negative, type="percent")

        else:
            # outputs["loss_grounding"] = torch.tensor(0.0).to(loss)
            grounding_loss = torch.tensor(0.0).to(outputs.logits)
            grounding_acc = 0.0
            grounding_acc_positive = 0.0
            grounding_acc_negative = 0.0
            grounding_acc_softmax = 0.0


        if not return_dict:
            # return tuple(v for v in outputs if v is not None)
            outputs =  tuple(v for v in outputs if v is not None)
            outputs = (loss, ) + outputs if self.training else outputs
        else:
            # if self.training:
            #     outputs.loss = loss
            outputs = CausalLMOutputWithPast(
                loss=loss if self.training else None,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            outputs = MyObjectDict(outputs.__dict__)
            if self.config.use_grounding_classifier:
                # outputs["grounding_logits"] = grounding_logits
                outputs["loss_grounding"] = grounding_loss
                outputs["grounding_acc"] = grounding_acc
                outputs["grounding_acc_positive"] = grounding_acc_positive
                outputs["grounding_acc_negative"] = grounding_acc_negative
                outputs["grounding_acc_softmax"] = grounding_acc_softmax
        return outputs

    def insert_position_ids(self, image_patch_input_indices, position_ids, len_scene_embeddings, scene_embedding_mask):
        print_once("insert_position_ids is called")
        # insert position_ids
        # input_ids is like [X, X, ..., X_img, ..., X_img, X_word, ..., X_word]
        # X ~ PAD token, X_img ~ image patch, X_word ~ word token
        # we want to insert before X_img
        # so we need to find the indices of X_img
        # image_patch_input_indices is like [-1, -1, ..., 0, 1, 2, ..., N, -1, -1, ...] where N is the number of image patches
        # i ~ i-th image patch, -1 ~ word token
        position_ids_concat_results = []
        for batch_idx in range(position_ids.shape[0]):
            number_scene_embeds = scene_embedding_mask[batch_idx].sum()
            position_id_scene_embeds = torch.cumsum(scene_embedding_mask[batch_idx], dim=0) - 1
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
            # insert position_ids
            position_ids_concat_result = torch.cat((
                position_ids[batch_idx, :image_patch_begin_idx], 
                # torch.arange(len_scene_embeddings, dtype=torch.long, device=position_ids.device), 
                position_id_scene_embeds,
                position_ids[batch_idx, image_patch_begin_idx:] + number_scene_embeds,
                ), 
                dim=0
            )
            position_ids_concat_results.append(position_ids_concat_result)
        position_ids = torch.stack(position_ids_concat_results, dim=0)
        return position_ids
    
    def get_image_patch_start_position(self, image_patch_input_indices):
        positions = []
        for batch_idx in range(image_patch_input_indices.shape[0]):
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
            positions.append(image_patch_begin_idx)
        positions = torch.tensor(positions, dtype=torch.long, device=image_patch_input_indices.device)
        return positions
    
    def get_image_patch_end_position(self, image_patch_input_indices):
        positions = []
        for batch_idx in range(image_patch_input_indices.shape[0]):
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][-1]
            positions.append(image_patch_begin_idx)
        positions = torch.tensor(positions, dtype=torch.long, device=image_patch_input_indices.device)
        return positions

    def insert_attention_mask(self, 
                                image_patch_input_indices, 
                                attention_mask, 
                                scene_attention_mask, 
                                scene_embed_mask=None,
                                use_3d=True, 
                                use_2d=True,
                                p_drop_2d=0.0,
                                p_drop_3d=0.0,
                                do_drop_2d_partial=False,
                                p_drop_2d_partial_alpha=2.0,
                                p_drop_2d_partial_beta=8.0,
                            ):
        if attention_mask is None: 
            return None
        attention_mask_concat_results = []
        new_scene_embed_masks = []

        if use_3d is False:
            print_once("WARNING: Not using 3D INPUTS!!!")
            scene_attention_mask = torch.zeros_like(scene_attention_mask)

        if p_drop_3d > 0.0:
            print_once(f"WARNING: dropping 3D tokens (sample-wise) with probability {p_drop_3d}!")
            dropped_samples = torch.rand(scene_attention_mask.shape[0], device=scene_attention_mask.device) < p_drop_3d
            scene_attention_mask[dropped_samples < p_drop_3d] = 0

        if use_2d is False:
            print_once("WARNING: Not using 2D INPUTS!!!")
            # mask off the 2D tokens
            # take out all the 2D tokens indices
            for batch_idx in range(attention_mask.shape[0]):
                # find where the image patches begins
                dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
                attention_mask[batch_idx, dst_indices] = 0

        if p_drop_2d > 0.0:
            print_once(f"WARNING: dropping 2D tokens (sample-wise) with probability {p_drop_2d}!")
            dropped_samples = torch.rand(attention_mask.shape[0], device=attention_mask.device) < p_drop_2d
            for batch_idx in range(attention_mask.shape[0]):
                # find where the image patches begins
                if dropped_samples[batch_idx]:
                    dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
                    attention_mask[batch_idx, dst_indices] = 0

        if do_drop_2d_partial:
            print_once(f"WARNING: dropping PARTIAL 2D tokens (in-sample) with probability ~ Beta({p_drop_2d_partial_alpha}, {p_drop_2d_partial_beta})!")
            drop_ratio = torch.distributions.Beta(p_drop_2d_partial_alpha, p_drop_2d_partial_beta).sample((attention_mask.shape[0],))
            for batch_idx in range(attention_mask.shape[0]):
                # find where the image patches begins
                dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
                attention_mask[batch_idx, dst_indices] = attention_mask[batch_idx, dst_indices] * (torch.rand_like(dst_indices.float()) > drop_ratio[batch_idx])

        # ic(attention_mask.shape, scene_attention_mask.shape)

        for batch_idx in range(attention_mask.shape[0]):
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
            # insert position_ids
            attention_mask_concat_result = torch.cat((
                attention_mask[batch_idx, :image_patch_begin_idx], 
                # torch.ones(len_scene_embeddings, dtype=torch.long, device=attention_mask.device), 
                scene_attention_mask[batch_idx],
                attention_mask[batch_idx, image_patch_begin_idx:]
                ), 
                dim=0
            )
            attention_mask_concat_results.append(attention_mask_concat_result)
            if scene_embed_mask is not None:
                new_scene_embed_mask = torch.cat((
                    torch.zeros(image_patch_begin_idx, dtype=torch.bool, device=scene_embed_mask.device),
                    scene_embed_mask[batch_idx],
                    torch.zeros(attention_mask.shape[1] - image_patch_begin_idx, dtype=torch.bool, device=scene_embed_mask.device),
                ), dim=0)
                new_scene_embed_masks.append(new_scene_embed_mask)
            
        attention_mask = torch.stack(attention_mask_concat_results, dim=0)
        if scene_embed_mask is not None:
            new_scene_embed_masks = torch.stack(new_scene_embed_masks, dim=0)
        else:
            new_scene_embed_masks = None
        return attention_mask, new_scene_embed_masks


    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
        scene_embeddings: torch.Tensor,
        scene_attention_mask: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings: Tensor of word embeddings. Shape: [b, s, h]
            continuous_embeddings:
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is
            shape [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
            indices in image_patch_input_indices for that batch element.
            image_patch_input_indices: Tensor of indices of the image patches in the input_ids tensor. Shape: [b, s]
            scene_embeddings: Tensor of scene embeddings. Shape: [b, num_scene_patches, h]
            attention_mask: Tensor of attention mask to process (insert 1 for scene tokens). Shape: [b, s]
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        len_scene_embeddings = scene_embeddings.shape[1]
        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # First, find the positions of all the non-negative values in image_patch_input_indices, those are the
            # positions in word_embeddings that we want to replace with content from continuous_embeddings.
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # Next look up those indices in image_patch_input_indices to find the indices in continuous_embeddings that we
            # want to use to replace the values in word_embeddings.
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # Check if we have more indices than embeddings. Note that we could have fewer indices if images got truncated.
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]

        # insert scene embeddings into word embeddings
        # word_embeddings is like [X, X, ..., X_img, ..., X_img, X_word, ..., X_word]
        # X ~ PAD token, X_img ~ image patch, X_word ~ word token
        # we want to insert before X_img
        # so we need to find the indices of X_img
        # image_patch_input_indices is like [-1, -1, ..., 0, 1, 2, ..., N, -1, -1, ...] where N is the number of image patches
        # i ~ i-th image patch, -1 ~ word token
        
        # FIXME: simply use scatter is even slower?
        # NOTE: we have additionally padded the sequence in get_sample_encoding(...,additional_tokens_length=N)
        # scene_embeds_indices = word_embeddings.new_zeros(scene_embeddings.shape[:2], dtype=torch.long) # [B, num_scene_patches]
        # for batch_idx in range(word_embeddings.shape[0]):
        #     # find where the image patches begins
        #     image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
        #     scene_embeds_indices[batch_idx] = torch.arange(image_patch_begin_idx - len_scene_embeddings, image_patch_begin_idx, device=word_embeddings.device)
        # === insert scene embeddings   
        # # output_embeddings = torch.where(mask, scene_embeddings, output_embeddings)
        # # scatter the scene embeddings to the correct positions
        # scene_embeds_indices = scene_embeds_indices.unsqueeze(-1).expand(-1, -1, scene_embeddings.shape[-1]) # [B, num_scene_patches, h]
        # output_embeddings = output_embeddings.scatter(1, scene_embeds_indices.unsqueeze(-1), scene_embeddings)
        
        # FIXME: how to do without loop
        concat_results = []
        attention_concat_results = []
        labels_concat_results = []
        position_ids_concat_results = []
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
            # insert scene embeddings
            concat_result = torch.cat((output_embeddings[batch_idx, :image_patch_begin_idx], scene_embeddings[batch_idx], output_embeddings[batch_idx, image_patch_begin_idx:]), dim=0)
            concat_results.append(concat_result)
            if attention_mask is not None:
                # insert attention mask
                # attention_concat_result = torch.cat((attention_mask[batch_idx, :image_patch_begin_idx], torch.ones(len_scene_embeddings, dtype=torch.long, device=attention_mask.device), attention_mask[batch_idx, image_patch_begin_idx:]), dim=0)
                attention_concat_result = torch.cat(
                    (attention_mask[batch_idx, :image_patch_begin_idx], 
                     scene_attention_mask[batch_idx], 
                     attention_mask[batch_idx, image_patch_begin_idx:]
                    )
                    , dim=0
                )
                # attention_mask[batch_idx] = attention_concat_result
                attention_concat_results.append(attention_concat_result)
            if labels is not None:
                # insert labels
                labels_concat_result = torch.cat((
                    labels[batch_idx, :image_patch_begin_idx], 
                    torch.full((len_scene_embeddings,), fill_value=-100, dtype=torch.long, device=labels.device),  # all is masked in CE
                    labels[batch_idx, image_patch_begin_idx:]
                    ), 
                    dim=0
                )
                labels_concat_results.append(labels_concat_result)
            if position_ids is not None:
                # insert position_ids
                position_ids_concat_result = torch.cat((
                    position_ids[batch_idx, :image_patch_begin_idx], 
                    torch.arange(len_scene_embeddings, dtype=torch.long, device=position_ids.device), 
                    position_ids[batch_idx, image_patch_begin_idx:]
                    ), 
                    dim=0
                )
                position_ids_concat_results.append(position_ids_concat_result)

        # TODO: use index_add_ to avoid loop!!
                


        output_embeddings = torch.stack(concat_results, dim=0)
        if attention_mask is not None:
            attention_mask = torch.stack(attention_concat_results, dim=0)
        if labels is not None:
            labels = torch.stack(labels_concat_results, dim=0)
        return output_embeddings, attention_mask, labels
        

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        image_patches=None,
        image_patches_indices=None,
        mnet_inputs: Optional[Tuple] = None,
        qformer_inputs: Optional[dict] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # FIXME: how to adapt to scene embeddings?
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids} 

        # also the 3D scene embeddings for the first time
        if mnet_inputs is not None and past_key_values is None:
            model_inputs["mnet_inputs"] = mnet_inputs

        # if qformer_inputs is not None and past_key_values is None:
        if qformer_inputs is not None: # must have whenever in generation
            model_inputs["qformer_inputs"] = qformer_inputs
        

        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_patches_indices": image_patches_indices, # always provide to insert position ids correctly # if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
                "focus_bbox": kwargs.get("focus_bbox", None),
                "target_object_indices": kwargs.get("target_object_indices", None),
                "related_object_bboxes": kwargs.get("related_object_bboxes", None),
            }
        )

        # use inspect to get the signature of the forward function, if any param is needed and in kwargs, pass it
        forward_signature = inspect.signature(self.forward)
        added_keys = set()
        exclude_params = {
            "input_ids", 
            "past_key_values", 
            "attention_mask", 
            "inputs_embeds", 
            "image_patches", "image_patches_indices", 
            "mnet_inputs", "qformer_inputs"}
        for param_name, param in forward_signature.parameters.items():
            if param.kind != inspect.Parameter.VAR_POSITIONAL and param.kind != inspect.Parameter.VAR_KEYWORD:
                # not *args, **kwargs
                if param_name in kwargs and param_name not in model_inputs and param_name not in exclude_params:
                    added_keys.add(param_name)
                    model_inputs[param_name] = kwargs[param_name]

        print_once(f"Added keys in generation: {sorted(added_keys)}")

        return model_inputs

    def save_pretrained(self, save_directory):
        # only save lora, if not, the model is frozen.
        if isinstance(self.fuyu, PeftModel):
            logger.info("Saving the LoRA.")
            self.fuyu.save_pretrained(save_directory)
        else:
            logger.info("Not saving the frozen Fuyu model.")

        # save all other params except "fuyu"
        state_dict = self.state_dict()
        all_other_params = {k: v for k, v in state_dict.items() if "fuyu" not in k}
        logger.info(f"Saving all other params: {all_other_params.keys()}")
        torch.save(all_other_params, os.path.join(save_directory, "other_params.pth"))

    def load_pretrained(self, save_directory):
        logger.info(f"Loading non-LLM checkpoint from {save_directory}")
        all_other_params = torch.load(os.path.join(save_directory, "other_params.pth"), map_location=torch.device("cpu"))
        # load all other params except "fuyu"
        message = self.load_state_dict(all_other_params, strict=False)
        logger.info(message)
        del all_other_params

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Copied from Persimmon (similar to GPT2)
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

