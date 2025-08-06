from transformers import FuyuProcessor, FuyuForCausalLM, AutoModelForCausalLM, FuyuProcessor, FuyuPreTrainedModel, FuyuConfig
from transformers.tokenization_utils_base import PaddingStrategy, TensorType, TruncationStrategy
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor, FuyuBatchFeature
import MinkowskiEngine as ME
from MinkowskiEngine import MinkowskiToDenseTensor
from models.minkunet import MinkUNet34C
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


# seems just not need, if we use for-loop + concat to insert scene embeddings
class Fuyu3DProcessor(FuyuProcessor):
    def __init__(self, image_processor, tokenizer, additional_scene_tokens_length =256):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.additional_scene_tokens_length = additional_scene_tokens_length # FIXME: how many we need? i.e. the output size of MinkUNet34C

    def __call__(
        self,
        text=None,
        images=None,
        add_special_tokens: bool = True,
        return_attention_mask: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_token_type_ids: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> "FuyuBatchFeature":
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to
        encode the text. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        FuyuImageProcessor's [`~FuyuImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `List[PIL.Image.Image]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """

        # --- Check input validity ---
        if not return_attention_mask:
            raise ValueError("`return_attention_mask=False` is not supported for this model.")
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be None.")
        if text is not None and images is None:
            logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            self.current_processor = self.tokenizer
            text_encoding = self.tokenizer(
                text=text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_offsets_mapping=return_offsets_mapping,
                return_token_type_ids=return_token_type_ids,
                return_length=return_length,
                verbose=verbose,
                return_tensors=return_tensors,
                **kwargs,
            )
            return text_encoding

        if text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            prompts = [[""]]
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]

        # --- Preprocess images using self.image_processor ---

        # FIXME - We hard code "pt" here because the rest of the processing assumes torch tensors
        image_encoding = self.image_processor.preprocess(images, return_tensors="pt")
        batch_images = image_encoding["images"]
        image_unpadded_heights = image_encoding["image_unpadded_heights"]
        image_unpadded_widths = image_encoding["image_unpadded_widths"]
        scale_factors = image_encoding["image_scale_factors"]
        self.subsequence_length = 1  # Each batch contains only one sequence.
        self.batch_size = len(batch_images)

        # --- Use self.tokenizer to get the ids of special tokens to insert into image ids ---

        image_placeholder_id = self.tokenizer("|SPEAKER|", add_special_tokens=False)["input_ids"][1]
        image_newline_id = self.tokenizer("|NEWLINE|", add_special_tokens=False)["input_ids"][1]
        tensor_batch_images = torch.stack([img[0] for img in batch_images]).unsqueeze(1)

        # --- Use self.image_processor again to obtain the full token ids and batch inputs ---
        all_encodings = []

        for prompt, scale_factor, image_unpadded_height, image_unpadded_width, tensor_batch_image in zip(
            prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, tensor_batch_images
        ):
            sample_encoding = self.get_sample_encoding(
                prompts=[prompt],
                scale_factors=[scale_factor],
                image_unpadded_heights=torch.tensor([image_unpadded_height]),
                image_unpadded_widths=torch.tensor([image_unpadded_width]),
                image_placeholder_id=image_placeholder_id,
                image_newline_id=image_newline_id,
                tensor_batch_images=tensor_batch_image.unsqueeze(0),
                # additional_tokens_length=self.additional_scene_tokens_length, # leave some space for scene encoding
                # NOTE: no need since we use concat to insert scene embeddings
            )
            all_encodings.append(sample_encoding)
        batch_encoding = self._left_pad_inputs_with_attention_mask(
            model_inputs=all_encodings, return_attention_mask=return_attention_mask
        )
        return FuyuBatchFeature(data=batch_encoding)


class Fuyu3DCausalLM(FuyuForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config.mnet_path = kwargs.get("mnet_path", "/scratch/generalvision/mowentao/ScanQA/weights.pth")
        self.config.freeze_mnet = kwargs.get("freeze_mnet", True)

        self.in_channels = kwargs.get("in_channels", 3)
        # 3D scene encoding
        self.mnet = MinkUNet34C(self.in_channels, 20)
        # out_channels = mnet.PLANES[7] * mnet.BLOCK.expansion
        out_channels_mnet = self.mnet.PLANES[7] * self.mnet.BLOCK.expansion
        print(f"MinkUNet out_channels: {out_channels_mnet}")
        self.spatial_patch_size = kwargs.get("spatial_patch_size", 24)
        self.pooling_method = kwargs.get("pooling_method", "max")
        if self.pooling_method == "max":
            self.mnet_pool = ME.MinkowskiMaxPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)
        elif self.pooling_method == "avg":
            self.mnet_pool = ME.MinkowskiAvgPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)
        elif self.pooling_method == "sum":
            self.mnet_pool = ME.MinkowskiSumPooling(kernel_size=self.spatial_patch_size, stride=self.spatial_patch_size, dimension=3)

        # self.to_dense = MinkowskiToDenseTensor()
        self.linear_3d = nn.Sequential(
            nn.LayerNorm(out_channels_mnet + 3),
            nn.Linear(out_channels_mnet + 3, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size)
        )
        self.load_mnet()
        # TODO: shall we add 3D-patch xyz position embedding?
    
    def load_mnet(self):
        # load pretrained MinkowskiNet
        print(f"Loading pretrained MinkowskiNet from {self.config.mnet_path}, freeze_mnet={self.config.freeze_mnet}")
        self.mnet = self.mnet.float() # MinkowskiNet is float32, not supporting half-precision
        model_dict = torch.load(self.config.mnet_path, map_location=torch.device('cpu'))
        self.mnet.load_state_dict(model_dict)
        if self.config.freeze_mnet:
            print.info("Freezing MinkowskiNet")
            self.mnet.eval()
            for p in self.mnet.parameters():
                p.requires_grad = False


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_patches: torch.Tensor = None,  # [batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mnet_inputs: Optional[Tuple] = None, # tuple of (coords, feats, labels)
        cached_coords_mask: Optional[torch.Tensor] = None,
        **kwargs, # hack to reduce the need to manually change the code 
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if mnet_inputs is not None:
            # disable autocast
            with torch.cuda.amp.autocast(enabled=False): # DO NOT use autocast here, since MinkowskiNet is float32
                # assert mnet_inputs is not None, "mnet_inputs must be provided for MinkowskiNet"
                coords, feats, point_labels = mnet_inputs
                # MinkowskiNet/3D scene encoding
                # FIXME: shall we skip if past_key_values is not None?
                out, pred = self.mnet(ME.SparseTensor(feats.float(), coords))
                # ic(out.shape)
                out = self.mnet_pool(out)
                # ic(out.shape)

                coords_batch, feats_batch = out.decomposed_coordinates_and_features # [B, N, 3], [B, N, C]
                # how to deal with inconsistent point/voxel numbers?
                # pad to the same number of points
                
            coords_batch = pad_sequence([coords for coords in coords_batch], batch_first=True, padding_value=0) # [B, N, 3]
            feats_batch = pad_sequence([feats for feats in feats_batch], batch_first=True, padding_value=0) # [B, N, C]
            # ic(coords_batch.shape, feats_batch.shape)
            coords_mask = torch.logical_not(torch.all(coords_batch == 0, dim=-1)) # [B, N]

            out = torch.cat((coords_batch, feats_batch), dim=-1) # [B, N, 3+C]

            # ic(out.shape)
            # print autocast status
            # print(f"autocast status: {torch.is_autocast_enabled()}")
            # print(out.dtype, self.linear_3d[0].weight.dtype)
            if out.dtype != self.linear_3d[0].weight.dtype:
                out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage

            out = self.linear_3d(out)
            len_scene_embeddings = out.shape[1]

            # cache for generation
            # self.cached_coords_mask = coords_mask.detach().clone()
            # self.cached_len_scene_embeddings = len_scene_embeddings
        else:
            out = None
            # coords_mask = self.cached_coords_mask.clone() # use cached coords mask from previous generation step
            coords_mask = cached_coords_mask
            # len_scene_embeddings = self.cached_len_scene_embeddings
            len_scene_embeddings = coords_mask.shape[1]
            ic(coords_mask.shape, len_scene_embeddings)
            print("No 3D scene encoding is used, this should be in the generation stage, at step > 1")

        # Fuyu LVLM
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
        
        # logger.info(f"attention_mask.shape: {attention_mask.shape}, coords_mask.shape: {coords_mask.shape}")
        # insert attention_mask, this is a must
        attention_mask = self.insert_attention_mask(
            image_patch_input_indices=image_patches_indices, 
            attention_mask=attention_mask,
            scene_attention_mask=coords_mask,
        )

        if inputs_embeds is None:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if image_patches is not None and past_key_values is None:
                patch_embeddings = [
                    self.vision_embed_tokens(patch.to(self.vision_embed_tokens.weight.dtype)).squeeze(0)
                    for patch in image_patches
                ]
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

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if not return_dict:
            return tuple(v for v in outputs if v is not None)
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

    def insert_attention_mask(self, image_patch_input_indices, attention_mask, scene_attention_mask):
        if attention_mask is None: 
            return None
        attention_mask_concat_results = []

        # ic(attention_mask.shape, scene_attention_mask.shape)
        logger.info(f"attention_mask.shape: {attention_mask.shape}, scene_attention_mask.shape: {scene_attention_mask.shape}")
        assert attention_mask.shape[0] == scene_attention_mask.shape[0], f"{attention_mask.shape=} {scene_attention_mask.shape=}, batch size must match!"
        input()

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
        # # insert scene embeddings   
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
            }
        )
        return model_inputs
