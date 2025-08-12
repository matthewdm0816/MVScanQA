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
from models.pnpp import PointNetPP
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutputWithPast
from peft import PeftModel
from icecream import ic
from typing import Optional, Tuple, Union, List
import os
from dataclasses import dataclass, asdict
from fuyu_utils import batch_calculate_reinforce_reward_labels, loss_reinforce
from lib.dataset import DC
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
from models.detector_Vote2Cap_DETR.detector import detector
from collections import namedtuple


import logging
logger = logging.getLogger(__name__)

def print_once(message):
    if not hasattr(print_once, "printed"):
        print_once.printed = True
        print(message)

def trim_objects(out, coords_mask, predicted_bbox_corners):
    # according to coords_mask, trim the objects
    # out ~ [B, N, H]
    # coords_mask ~ [B, N]
    # predicted_bbox_corners ~ [B, N, 8, 3]
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
        return self[name]
    
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
        # elif adapter_type == "mixed-ffn":
        #     # self.linear_3d = nn.Sequential(
        #     #     nn.LayerNorm(self.config.out_channels_mnet),
        #     #     nn.Linear(self.config.out_channels_mnet, self.config.hidden_size),
        #     #     nn.GELU(),
        #     #     nn.Dropout(0.1),
        #     #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     # )
        #     assert isinstance(self.config.out_channels_mnet, list), "out_channels_mnet must be a list"
        #     self.linear_3d = nn.ModuleList([
        #             nn.Sequential(
        #             nn.LayerNorm(d),
        #             nn.Linear(d, self.config.hidden_size),
        #             nn.GELU(),
        #             nn.Dropout(0.1),
        #             nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #         ) for d in self.config.out_channels_mnet
        #     ])
        #     self.linear_focus_bbox = nn.Sequential(
        #         nn.Linear(6, self.config.hidden_size),
        #         nn.GELU(),
        #         nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     )
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
        self.fuyu = FuyuForCausalLM.from_pretrained(**pretrained_args)
        
        # inherit from FuyuForCausalLM
        self.config = self.fuyu.config
        self.padding_idx = self.fuyu.padding_idx
        self.vocab_size = self.fuyu.vocab_size
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

        self.config.predict_frame_params = kwargs.get("predict_frame_params", False)
        self.config.coeff_frame_params = kwargs.get("coeff_frame_params", 0.1)
        if self.config.predict_frame_params:
        # 2x4x4 for camera intrinsics, pose | axis alignments as input
            # self.frame_params_head = nn.Sequential(
            #     nn.Linear(self.config.hidden_size + 4*4, self.config.hidden_size * 4),
            #     nn.GELU(),
            #     nn.Linear(self.config.hidden_size * 4, self.config.hidden_size),
            #     nn.GELU(),
            #     nn.Linear(self.config.hidden_size, 2*4*4),
            # )
            self.frame_params_head = nn.Linear(self.config.hidden_size + 4*4, 2*4*4)
        
        ## 2D & 3D MASK SETTINGs
        self.config.p_drop_2d = kwargs.get("p_drop_2d", 0.0) # prob to drop ALL 2D tokens
        self.config.p_drop_3d = kwargs.get("p_drop_3d", 0.0) # prob to drop ALL 3D tokens

        self.config.do_drop_2d_partial = kwargs.get("do_drop_2d_partial", False) # do drop PARTIAL 2D tokens
        # if we do, the drop ratio ~ Beta(p_drop_2d_partial_alpha, p_drop_2d_partial_beta), mean = alpha / (alpha + beta)
        self.config.p_drop_2d_partial_alpha = kwargs.get("p_drop_2d_partial_alpha", 2.0) # prob to drop PARTIAL 2D tokens
        self.config.p_drop_2d_partial_beta = kwargs.get("p_drop_2d_partial_beta", 8.0) # prob to drop PARTIAL 2D tokens

        self.related_object_embedding = nn.Parameter(torch.zeros(1, self.config.hidden_size))
        nn.init.kaiming_normal_(self.related_object_embedding)
        # self.related_object_mlp = nn.Sequential(
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(self.config.hidden_size, self.config.hidden_size),
        # )
        self.related_object_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.out_channels_mnet),
            # nn.GELU(),
            # nn.Linear(self.config.out_channels_mnet * 2, self.config.out_channels_mnet),
        )

        self.config.choose_related_object = kwargs.get("choose_related_object", False)
        self.config.trim_objects = kwargs.get("trim_objects", True)


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

            # if self.pre_proj is not None:
            if self.config.pc_tokenizer_type == "merged-frozen":
                out = torch.cat([proj(x) for proj, x in zip(self.pre_proj, out)], dim=1) # [B, N1, H] + [B, N2, H] + ... -> [B, N, H]
                coords_mask = torch.cat([mask for mask in coords_mask], dim=1)
                if predicted_bbox_corners is not None:
                    predicted_bbox_corners = torch.cat([bbox_corners for bbox_corners in predicted_bbox_corners], dim=1)

            print_once(f"predicted_bbox_corners: {predicted_bbox_corners.shape if predicted_bbox_corners is not None else None}")
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
                # print(f"Beam search detected, {beam_size=}")

            
            linear_3d_weight_dtype = self.linear_3d[0].weight.dtype if isinstance(self.linear_3d, nn.Sequential) else self.linear_3d.weight.dtype
            if out.dtype != linear_3d_weight_dtype:
                out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage
            
            out = self.linear_3d(out)
            if self.config.adapter_type == "upsampler":
                # [B, L, H*upsample_ratio] -> [B, L*upsample_ratio, H]
                out = out.view(out.shape[0], -1, self.config.hidden_size) # [B, L * upsample_ratio, H]
                # 0, ..., upsample_ratio-1 ~ 0-th token, mask=coords_mask[..., 0]
                # upsample_ratio, ..., 2*upsample_ratio-1 ~ 1-th token, mask=coords_mask[..., 1]
                # thus repeat coords_mask with upsample_ratio times

                coords_mask = coords_mask.repeat_interleave(self.upsample_ratio, dim=1)

            if is_in_beam_search:
                out = out.repeat_interleave(beam_size, dim=0)
                coords_mask = coords_mask.repeat_interleave(beam_size, dim=0)
            
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
                # len_scene_embeddings += self.num_think_tokens
            
            len_scene_embeddings = out.shape[1]
            self.cached_coords_mask = coords_mask.detach().cpu().clone().numpy() # if in beam search, this is repeated
            self.cached_len_scene_embeddings = len_scene_embeddings
        else:
            coords_mask: np.ndarray = self.cached_coords_mask.copy() # use cached coords mask from previous generation step
            coords_mask = torch.from_numpy(coords_mask).to(input_ids.device)
            len_scene_embeddings = self.cached_len_scene_embeddings

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                beam_size = input_ids.shape[0] // coords_mask.shape[0]

            # out = None
            # make deepspeed zero3 stage happy: avoid unused parameter
            # --- these computation is pure useless
            # out = torch.zeros(1, 1, self.config.out_channels_mnet, dtype=torch.float32, device=input_ids.device)
            # linear_3d_weight_dtype = self.linear_3d[0].weight.dtype if isinstance(self.linear_3d, nn.Sequential) else self.linear_3d.weight.dtype
            # if out.dtype != linear_3d_weight_dtype:
            #     out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage

            # out = self.linear_3d(out)
            # if self.use_focus_bbox:
            #     if focus_bbox is None:
            #         focus_bbox = torch.zeros(out.shape[0], 1, 6, dtype=out.dtype, device=out.device)
                
            #     focus_bbox = self.linear_focus_bbox(focus_bbox.to(out.dtype))
            # --- end of useless computation
            out = None
            
            # NOTE: this does not work for beam-search!!!
            # ic(coords_mask.shape, len_scene_embeddings)
            # ic("No 3D scene encoding is used, this should be in the generation stage, at step > 1")

        # detect beam search (in this occasion, mnet_inputs is for one batch only, while input_ids is duplicated for beam search)
        # and pad coords_mask and out accordingly
        # if is_in_beam_search:
        #     coords_mask = coords_mask.repeat_interleave(beam_size, dim=0)
        

        # Fuyu LVLM
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
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

        seq_length += len_scene_embeddings # to generate appropriate position_ids including scene embeddings

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        else:
            # add len_scene_embeddings to position_ids to generate appropriate position_ids including scene embeddings
            position_ids = position_ids + len_scene_embeddings 
            # NOTE: only execute when past_key_values is None, otherwise position_ids is already [1] length
            if past_key_values is None:
                position_ids = self.insert_position_ids(
                    image_patch_input_indices=image_patches_indices, 
                    position_ids=position_ids, 
                    len_scene_embeddings=len_scene_embeddings
                )
        
        # insert scene attention_mask, this is a must
        attention_mask = self.insert_attention_mask(
            image_patch_input_indices=image_patches_indices, 
            attention_mask=attention_mask,
            scene_attention_mask=coords_mask,
            use_3d=self.config.use_3d,
            use_2d=self.config.use_2d,
            p_drop_2d=self.config.p_drop_2d if self.training else 0.0,
            p_drop_3d=self.config.p_drop_3d if self.training else 0.0,
            do_drop_2d_partial=self.config.do_drop_2d_partial if self.training else False,
            p_drop_2d_partial_alpha=self.config.p_drop_2d_partial_alpha,
            p_drop_2d_partial_beta=self.config.p_drop_2d_partial_beta,
        )


        if inputs_embeds is None:
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
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels) # [B * N]
                # loss = loss.view(_label_shape[0], -1) # [B, N]
                # # unmasked_tokens = (labels != -100).sum(dim=-1)
                # # sum through tokens, mean through batch
                # loss = loss.sum(dim=-1).mean()
                # print(loss)

        if self.config.predict_frame_params and frame_poses is not None and frame_intrinsics is not None:
            # frame_params_mask ~ those frame_poses not I(4) 
            last_hidden_state = outputs.hidden_states[-1] # [B, L, H]
            # logger.info(f"last_hidden_state: {last_hidden_state.shape}")

            frame_params_mask = (frame_poses != torch.eye(4, device=last_hidden_state.device).unsqueeze(0)).any(dim=-1).any(dim=-1)
            total_frame_params = frame_params_mask.sum() # total samples to predict frame parameters
            # logger.info(f"frame_params_mask: {total_frame_params} / {frame_params_mask.shape[0]}")
            if total_frame_params == 0:
                loss_frame = torch.tensor(0.0).to(last_hidden_state)
                loss_frame_scene = torch.tensor(0.0).to(last_hidden_state)
                predicted_frame_pose = torch.eye(4, device=last_hidden_state.device).unsqueeze(0).expand(last_hidden_state.shape[0], -1, -1)
                predicted_frame_intrinsics = torch.eye(4, device=last_hidden_state.device).unsqueeze(0).expand(last_hidden_state.shape[0], -1, -1)
                predicted_frame_pose_scene = torch.eye(4, device=last_hidden_state.device).unsqueeze(0).expand(last_hidden_state.shape[0], -1, -1)
                predicted_frame_intrinsics_scene = torch.eye(4, device=last_hidden_state.device).unsqueeze(0).expand(last_hidden_state.shape[0], -1, -1)
            else:   
                # pooled_hidden_state = last_hidden_state.mean(dim=1) # [B, H]
                # scene_end_position = self.get_image_patch_start_position(image_patches_indices) - 1

                # NOTE: after seeing image (view) and scene, we predict the frame parameters
                #   note that the tokens are [scene embeddings, image embeddings, word embeddings]

                image_end_position = self.get_image_patch_end_position(image_patches_indices) # [B]
                image_end_position += len_scene_embeddings + int(self.use_focus_bbox) # add scene embeddings and focus bbox embeddings
                pooled_hidden_state = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), image_end_position] # [B, H]
                axis_alignments = axis_alignments.view(-1, 16) # [B, 4, 4] -> [B, 16]
                predicted_frame_params = self.frame_params_head(torch.cat((pooled_hidden_state, axis_alignments), dim=-1)) # [B, 2*4*4]
                predicted_frame_pose = predicted_frame_params[:, :16]
                predicted_frame_intrinsics = predicted_frame_params[:, 16:]

                loss_fct_params = nn.MSELoss()
                # loss_fct_params = nn.HuberLoss()
                loss_frame_pose = loss_fct_params(predicted_frame_pose[frame_params_mask], frame_poses[frame_params_mask].view(-1, 16))
                loss_frame_intrinsics = loss_fct_params(predicted_frame_intrinsics[frame_params_mask], frame_intrinsics[frame_params_mask].view(-1, 16))
                loss_frame = loss_frame_pose + loss_frame_intrinsics

                # scene_end_position = self.get_image_patch_start_position(image_patches_indices) - 1 + len_scene_embeddings
                # pooled_hidden_state_scene = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), scene_end_position] # [B, H]
                # predicted_frame_params_scene = self.frame_params_head(torch.cat((pooled_hidden_state_scene, axis_alignments), dim=-1)) # [B, 2*4*4]
                # predicted_frame_pose_scene = predicted_frame_params_scene[:, :16]
                # predicted_frame_intrinsics_scene = predicted_frame_params_scene[:, 16:]

                # loss_frame_pose_scene = loss_fct_params(predicted_frame_pose_scene[frame_params_mask], frame_poses[frame_params_mask].view(-1, 16))
                # loss_frame_intrinsics_scene = loss_fct_params(predicted_frame_intrinsics_scene[frame_params_mask], frame_intrinsics[frame_params_mask].view(-1, 16))
                # loss_frame_scene = loss_frame_pose_scene + loss_frame_intrinsics_scene


            # loss = loss + (loss_frame + loss_frame_scene) * self.config.coeff_frame_params
            loss = loss + loss_frame * self.config.coeff_frame_params


        # print(return_dict)
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
            if self.config.predict_frame_params and frame_poses is not None and frame_intrinsics is not None:
                outputs["predicted_frame_pose"] = predicted_frame_pose.view(-1, 4, 4)
                outputs["predicted_frame_intrinsics"] = predicted_frame_intrinsics.view(-1, 4, 4)
                outputs["loss_frame"] = loss_frame
                # outputs["predicted_frame_pose_scene"] = predicted_frame_pose_scene.view(-1, 4, 4)
                # outputs["predicted_frame_intrinsics_scene"] = predicted_frame_intrinsics_scene.view(-1, 4, 4)
                # outputs["loss_frame_scene"] = loss_frame_scene
                outputs = MyObjectDict(outputs.__dict__)
        return outputs

    def insert_position_ids(self, image_patch_input_indices, position_ids, len_scene_embeddings):
        # insert position_ids
        # input_ids is like [X, X, ..., X_img, ..., X_img, X_word, ..., X_word]
        # X ~ PAD token, X_img ~ image patch, X_word ~ word token
        # we want to insert before X_img
        # so we need to find the indices of X_img
        # image_patch_input_indices is like [-1, -1, ..., 0, 1, 2, ..., N, -1, -1, ...] where N is the number of image patches
        # i ~ i-th image patch, -1 ~ word token
        position_ids_concat_results = []
        for batch_idx in range(position_ids.shape[0]):
            # find where the image patches begins
            image_patch_begin_idx = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0][0]
            # insert position_ids
            position_ids_concat_result = torch.cat((
                position_ids[batch_idx, :image_patch_begin_idx], 
                torch.arange(len_scene_embeddings, dtype=torch.long, device=position_ids.device), 
                position_ids[batch_idx, image_patch_begin_idx:]
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
        attention_mask = torch.stack(attention_mask_concat_results, dim=0)
        return attention_mask


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

        if qformer_inputs is not None and past_key_values is None:
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
                "focus_bbox": kwargs.get("focus_bbox", None)
            }
        )
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


class LLM3DProcessorWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, images, padding="max_length", truncation=True, return_tensors="pt", **kwargs):
        # NOTE: this is a hack to make the processor compatible with the tokenizer
        #    we ignore the images here, since it is a pure text processor
        return self.tokenizer(
            text=text,
            padding="longest",
            truncation=truncation,
            return_tensors=return_tensors,
            max_length=512,
            **kwargs,
        )
    


class LLM3DCausalLM(PreTrainedModel, PCTokenizerAdapterMixin):
# class LLM3DCausalLM(nn.Module, GenerationMixin, PCTokenizerAdapterMixin):
    def __init__(self, **kwargs):
        llm_type = kwargs.get("llm_type", "mistral")
        if llm_type == "mistral":
            config = MistralConfig.from_pretrained(kwargs.get("model_id"))
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")
        
        super().__init__(config)

        # print kwargs without vocab
        print({k: v for k, v in kwargs.items() if k != "vocab"})
        # self.config = MyObjectDict()
        
        self.config.llm_type = kwargs.get("llm_type", "mistral")
        if self.config.llm_type == "mistral":
            self.llm = MistralForCausalLM.from_pretrained(kwargs.get("model_id"), torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unknown LLM type: {kwargs.get('llm_type')}")
        
        self.fuyu = self.llm # alias for the LLM
        
        self.config = self.llm.config
        self.config.iosa_threshold = kwargs.get("iosa_threshold", 0.25)

        # PC tokenizer options
        self.config.mnet_path = kwargs.get("mnet_path", "/scratch/generalvision/mowentao/ScanQA/weights.pth")
        self.config.pnpp_path = kwargs.get("pnpp_path", "...")
        self.config.vote2cap_detr_path = kwargs.get("vote2cap_detr_path", "...")
        self.config.freeze_vote2cap_detr = kwargs.get("freeze_vote2cap_detr", True)
        self.config.freeze_mnet = kwargs.get("freeze_mnet", True)
        self.config.freeze_pnpp = kwargs.get("freeze_pnpp", True)

        pc_tokenizer_type = kwargs.get("pc_tokenizer_type", "minkowski")
        self.config.pc_tokenizer_type = pc_tokenizer_type
        self.config.in_channels = kwargs.get("in_channels", 3)
        self.config.spatial_patch_size = kwargs.get("spatial_patch_size", 24)
        self.config.pooling_method = kwargs.get("pooling_method", "max")
        self.config.vote2cap_return_type = kwargs.get("vote2cap_return_type", "enc_features")
        self.config.frozen_in_channels = kwargs.get("frozen_in_channels", 256)
        self.config.merged_frozen_in_channels = kwargs.get("merged_frozen_in_channels", [256, 256])


        # Adapter options
        self.config.adapter_type = kwargs.get("adapter_type", "ffn")
        self.adater_type = self.config.adapter_type
        self.config.num_query_tokens = kwargs.get("num_query_tokens", 128)
        self.config.upsample_ratio = kwargs.get("upsample_ratio", 2)
        self.config.use_focus_bbox = kwargs.get("use_focus_bbox", False)
        self.config.pretrained_qformer = kwargs.get("pretrained_qformer", None)
        self.config.qformer_num_hidden_layers = kwargs.get("qformer_num_hidden_layers", 12)

        
        self._init_pc_tokenizer()

        self._init_adapter()
        
        self.use_focus_bbox = kwargs.get("use_focus_bbox", False)


    def forward(
        self,
        input_ids: torch.LongTensor = None,
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
        **kwargs, # hack to reduce the need to manually change the code 
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        is_in_beam_search = False

        # --- 3D PREFIXES
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

            

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                # NOTE: in Minkowski case, the input SparseTensor can't be automatically expanded
                #       so we need to repeat the coords_mask and out manually
                #       in PointNet++ case, the input is auto expanded, so the code will not reach here
                beam_size = input_ids.shape[0] // coords_mask.shape[0]
                # print(f"Beam search detected, {beam_size=}")

            
            linear_3d_weight_dtype = self.linear_3d[0].weight.dtype if isinstance(self.linear_3d, nn.Sequential) else self.linear_3d.weight.dtype
            if out.dtype != linear_3d_weight_dtype:
                out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage
            
            out = self.linear_3d(out)
            if self.config.adapter_type == "upsampler":
                # [B, L, H*upsample_ratio] -> [B, L*upsample_ratio, H]
                out = out.view(out.shape[0], -1, self.config.hidden_size) # [B, L * upsample_ratio, H]
                # 0, ..., upsample_ratio-1 ~ 0-th token, mask=coords_mask[..., 0]
                # upsample_ratio, ..., 2*upsample_ratio-1 ~ 1-th token, mask=coords_mask[..., 1]
                # thus repeat coords_mask with upsample_ratio times

                coords_mask = coords_mask.repeat_interleave(self.upsample_ratio, dim=1)

            if is_in_beam_search:
                out = out.repeat_interleave(beam_size, dim=0)
                coords_mask = coords_mask.repeat_interleave(beam_size, dim=0)
            
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
            

            # if self.num_think_tokens > 0:
            #     think_tokens = self.think_tokens.expand(out.shape[0], -1, -1)
            #     out = torch.cat((out, think_tokens), dim=1)
            #     coords_mask = torch.cat((coords_mask, torch.ones(coords_mask.shape[0], self.num_think_tokens, dtype=torch.bool, device=coords_mask.device)), dim=1)
                # len_scene_embeddings += self.num_think_tokens
            
            len_scene_embeddings = out.shape[1]
            self.cached_coords_mask = coords_mask.detach().cpu().clone().numpy() # if in beam search, this is repeated
            self.cached_len_scene_embeddings = len_scene_embeddings
        else:
            coords_mask: np.ndarray = self.cached_coords_mask.copy() # use cached coords mask from previous generation step
            coords_mask = torch.from_numpy(coords_mask).to(input_ids.device)
            len_scene_embeddings = self.cached_len_scene_embeddings

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                beam_size = input_ids.shape[0] // coords_mask.shape[0]

            out = None
            
            # NOTE: this does not work for beam-search!!!
            # ic(coords_mask.shape, len_scene_embeddings)
            # ic("No 3D scene encoding is used, this should be in the generation stage, at step > 1")
        # --- END 3D PREFIXES, get `out, coords_mask, len_scene_embeddings`
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        # ---- batch x (ntoken + nword) x n_embd
        if inputs_embeds is None:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long)
        # position_ids = position_ids if position_ids is not None else torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        # logger.info(f"len_scene_embeddings: {len_scene_embeddings}")

        # prepend the 3D scene embeddings
        if out is not None:
            # in case of generation after the first step, out is included in past_key_values
            inputs_embeds = torch.cat((out, inputs_embeds), dim=1)
        attention_mask = torch.cat((coords_mask, attention_mask), dim=1)
        # position_ids_scene = torch.arange(len_scene_embeddings, device=input_ids.device).unsqueeze(0).repeat(position_ids.shape[0], 1)
        # position_ids = torch.cat((
        #     torch.arange(len_scene_embeddings, device=input_ids.device).unsqueeze(0).repeat(), position_ids), dim=1
        # )
        # position_ids = torch.cat((position_ids_scene, position_ids), dim=1)
        
        
        seq_length += len_scene_embeddings # to generate appropriate position_ids including scene embeddings
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        else:
            # add len_scene_embeddings to position_ids to generate appropriate position_ids including scene embeddings
            position_ids = position_ids + len_scene_embeddings 
            # NOTE: only execute when past_key_values is None, otherwise position_ids is already [1] length
            if past_key_values is None:
                # position_ids = self.insert_position_ids(
                #     image_patch_input_indices=image_patches_indices, 
                #     position_ids=position_ids, 
                #     len_scene_embeddings=len_scene_embeddings
                # )
                position_ids_scene = torch.arange(len_scene_embeddings, device=input_ids.device).unsqueeze(0).repeat(position_ids.shape[0], 1)
                position_ids = torch.cat((position_ids_scene, position_ids), dim=1)


        if labels is not None:
            labels = torch.cat((torch.full((labels.shape[0], len_scene_embeddings), -100, dtype=torch.long, device=labels.device), labels), dim=1)
        
        # logger.info(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, position_ids: {position_ids.shape}")
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm.model(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # logger.info(f"outputs: {outputs}")

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)
        logits = logits.float()

        # logger.info(f"logits: {logits.shape}, labels: {labels.shape if labels is not None else None}")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        # image_patches=None,
        # image_patches_indices=None,
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

        if qformer_inputs is not None and past_key_values is None:
            model_inputs["qformer_inputs"] = qformer_inputs
        

        # if image_patches_indices is not None:
        #     model_inputs["image_patches_indices"] = image_patches_indices

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "image_patches_indices": image_patches_indices, # always provide to insert position ids correctly # if past_key_values is None else None,
                # "image_patches": image_patches if past_key_values is None else None,
                "focus_bbox": kwargs.get("focus_bbox", None)
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
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
        all_other_params = torch.load(os.path.join(save_directory, "other_params.pth"))
        # load all other params except "fuyu"
        message = self.load_state_dict(all_other_params, strict=False)
        logger.info(message)
        del all_other_params