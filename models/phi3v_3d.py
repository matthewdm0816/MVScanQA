from typing import Optional, Tuple, Union, List
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.modeling_phi3_v import Phi3VForCausalLM
from models.configuration_phi3_v import Phi3VConfig
from models.fuyu_3d import PCTokenizerAdapterMixin, print_once, trim_objects

from fuyu_align_utils import (
    calculate_in_view_objects,
    calculate_related_objects,
)

from transformers.tokenization_utils_base import PaddingStrategy, TensorType, TruncationStrategy
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers import AutoModelForCausalLM, PreTrainedModel, GenerationMixin
from peft import PeftModel
from icecream import ic

from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()

import logging
logger = logging.getLogger(__name__)


class Phi3V3DCausalLM(PreTrainedModel, PCTokenizerAdapterMixin):
    _supports_flash_attn_2 = True

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id")
        logger.info(f"Loading Phi3V3DCausalLM from {model_id}")
        config = Phi3VConfig.from_pretrained(model_id, _attn_implementation='flash_attention_2')
        super().__init__(config)

        print({k: v for k, v in kwargs.items() if k != "vocab"})
        
        self.llm = Phi3VForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.fuyu = self.llm  # alias for the LLM
        
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

        self.config.trim_objects = kwargs.get("trim_objects", True)

        self.config.use_object_index_embedding = kwargs.get("use_object_index_embedding", False)
        if self.config.use_object_index_embedding:
            assert self.config.trim_objects, "we assume objects are no-masked starting from 0 to some N"
            # init object index embedding
            self.object_index_embedding = nn.Embedding(512, self.config.hidden_size)

        self.config.use_object_textual_index = kwargs.get("use_object_textual_index", False)
        if self.config.use_object_textual_index:
            # from tokenizer, find the digit token ids
            self.text_tokenizer = kwargs.get("text_tokenizer", None)
            assert self.text_tokenizer is not None, "text_tokenizer must be provided for use_object_textual_index"

            logger.info(f"Adding new tokens embeddings for object textual index")
            self.fuyu.language_model.resize_token_embeddings(len(self.text_tokenizer))
            self.object_index_embedding_start = self.text_tokenizer.convert_tokens_to_ids("<OBJ0>")
            self.object_index_embedding_end = self.text_tokenizer.convert_tokens_to_ids("<OBJ511>")

    @property
    def vocab_size(self):
        return self.llm.config.vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        focus_bbox: Optional[torch.FloatTensor] = None,
        focus_bbox_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mnet_inputs: Optional[Tuple] = None,
        qformer_inputs: Optional[dict] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        frame_caption_mask: Optional[torch.BoolTensor] = None,
        frame_intrinsics: Optional[torch.FloatTensor] = None,
        frame_poses: Optional[torch.FloatTensor] = None,
        axis_alignments: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        is_in_beam_search = False

        # Process 3D inputs
        if mnet_inputs is not None:
            pc_tokenizer_output = self.pc_tokenizer(mnet_inputs) if self.config.pc_tokenizer_type != "frozen" else mnet_inputs
            out, coords_mask = pc_tokenizer_output[:2]
            predicted_bbox_corners = pc_tokenizer_output[2] if len(pc_tokenizer_output) > 2 else None

            is_in_beam_search = input_ids.shape[0] != coords_mask.shape[0]
            if is_in_beam_search:
                beam_size = input_ids.shape[0] // coords_mask.shape[0]
                out = out.repeat_interleave(beam_size, dim=0)
                coords_mask = coords_mask.repeat_interleave(beam_size, dim=0)

            if self.config.pc_tokenizer_type == "frozen" and self.config.trim_objects:
                # trim objects to shrink length
                out, coords_mask, predicted_bbox_corners = trim_objects(out, coords_mask, predicted_bbox_corners)


            linear_3d_weight_dtype = self.linear_3d[0].weight.dtype if isinstance(self.linear_3d, nn.Sequential) else self.linear_3d.weight.dtype
            if out.dtype != linear_3d_weight_dtype:
                out = out.to(self.linear_3d[0].weight.dtype) # sometimes autocast does not work in inference stage

            out = self.linear_3d(out.to(self.linear_3d[0].weight.dtype))

            # add object index embedding
            if self.config.use_object_index_embedding:
                object_indices = torch.arange(out.shape[1], device=out.device).unsqueeze(0).expand(out.shape[0], -1)
                object_indices = self.object_index_embedding(object_indices)
                out = out + object_indices

            if self.config.use_object_textual_index:
                # object_indices = [str(i) for i in range(out.shape[1])]
                object_indices = [f"<OBJ{i}>" for i in range(out.shape[1])]
                object_indices = [self.text_tokenizer.encode(obj, add_special_tokens=False) for obj in object_indices]
                # we assume all object index string is tokenized into same length of tokens
                # object_indices = self.text_tokenizer.convert_tokens_to_ids(object_indices)
                len_obj_tokens = len(object_indices[0])
                print_once(f"object token length: {len_obj_tokens}") # if added tokens, shall be 1
                all_object_indices = torch.tensor(sum(object_indices, []), device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N_objects * len_obj_tokens]
                # object_indices = torch.tensor(object_indices, device=out.device).unsqueeze(0).expand(out.shape[0], -1) # [B, N_objects]
                object_index_embeds = self.fuyu.language_model.get_input_embeddings()(all_object_indices).to(out) # [B, N_objects * len_obj_tokens, H]
                # added_object_embeds = self.fuyu.language_model.get_input_embeddings().weight[self.object_index_embedding_start:self.object_index_embedding_end+1]


                coords_mask = coords_mask.repeat_interleave((len_obj_tokens + 1), dim=1) # [B, N_objects] -> [B, N_objects * (len_obj_tokens + 1)]

                # Interleave object indices with out
                interleaved = torch.zeros((out.shape[0], out.shape[1] * (len_obj_tokens + 1), out.shape[2]), device=out.device, dtype=out.dtype) # [B, N_objects * (len_obj_tokens + 1), H]
                # interleaved[:, 0::2] = object_index_embeds
                # interleaved[:, 1::2] = out
                # interleaved[:, 0::(len_obj_tokens + 1)] = object_index_embeds
                # interleaved[:, len_obj_tokens::(len_obj_tokens + 1)] = out
                scene_embed_mask = torch.zeros(out.shape[0], out.shape[1] * (len_obj_tokens + 1), dtype=torch.bool, device=out.device)
                scene_embed_mask[:, len_obj_tokens::(len_obj_tokens + 1)] = True # place to put scene embeddings
                
                interleaved[scene_embed_mask] = out
                interleaved[~scene_embed_mask] = object_index_embeds

                # Update out with the interleaved features
                out = interleaved
            
            if self.config.adapter_type == "upsampler":
                out = out.view(out.shape[0], -1, self.config.hidden_size)
                coords_mask = coords_mask.repeat_interleave(self.upsample_ratio, dim=1)

            if self.use_focus_bbox:
                focus_bbox = focus_bbox.to(out.dtype).unsqueeze(1) if focus_bbox is not None else torch.zeros(out.shape[0], 1, 6, dtype=out.dtype, device=out.device)
                focus_bbox = self.linear_focus_bbox(focus_bbox)
                out = torch.cat((out, focus_bbox), dim=1)
                coords_mask = torch.cat((coords_mask, (focus_bbox_mask.unsqueeze(1) if focus_bbox_mask is not None else torch.ones(coords_mask.shape[0], 1, dtype=torch.bool, device=coords_mask.device))), dim=1)

            if self.config.adapter_type in ["qformer", "moe-qformer"]:
                # Process with QFormer
                query_attention_mask = torch.ones(out.shape[0], self.config.num_query_tokens, dtype=torch.bool, device=out.device)
                query_attention_mask = torch.cat((query_attention_mask, qformer_inputs["attention_mask"]), dim=1)
                
                out = self.qformer(
                    input_ids=qformer_inputs["input_ids"],
                    attention_mask=query_attention_mask,
                    query_embeds=self.qformer_query_tokens.expand(out.shape[0], -1, -1),
                    encoder_hidden_states=out,
                    encoder_attention_mask=coords_mask,
                    return_dict=True,
                ).last_hidden_state
                out = self.qformer_to_language(out)
                coords_mask = torch.ones(*out.shape[:2], dtype=torch.bool, device=out.device)

            len_scene_embeddings = out.shape[1]
            self.cached_coords_mask = coords_mask.detach().cpu().clone().numpy()
            self.cached_len_scene_embeddings = len_scene_embeddings
        else:
            coords_mask = torch.from_numpy(self.cached_coords_mask.copy()).to(input_ids.device)
            len_scene_embeddings = self.cached_len_scene_embeddings
            out = None

        # Prepare inputs for the LLM
        if inputs_embeds is None:
            # inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            if pixel_values is not None and image_sizes is not None:
                assert self.llm.vision_embed_tokens is not None, "Vision embedding layer is not defined"
                inputs_embeds = self.llm.vision_embed_tokens(input_ids, pixel_values=pixel_values, image_sizes=image_sizes)
            else:
                inputs_embeds = self.llm.embed_tokens(input_ids)
        
        if out is not None:
            inputs_embeds = torch.cat((out, inputs_embeds), dim=1)
        
        attention_mask = torch.cat((coords_mask, attention_mask), dim=1) if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.long)
        
        seq_length = inputs_embeds.shape[1]
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        if position_ids is None:
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        
        if labels is not None:
            labels = torch.cat((torch.full((labels.shape[0], len_scene_embeddings), -100, dtype=torch.long, device=labels.device), labels), dim=1)


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True
        use_cache = True
        return_dict = True

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            if pixel_values is not None and image_sizes is not None:
                assert self.vision_embed_tokens is not None, "Vision embedding layer is not defined"
                inputs_embeds = self.vision_embed_tokens(input_ids, pixel_values=pixel_values, image_sizes=image_sizes)
            else:
                inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
        # Forward pass through the LLM
        outputs = self.llm.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

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
        mnet_inputs: Optional[Tuple] = None,
        qformer_inputs: Optional[dict] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # Prepare model inputs
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "inputs_embeds": inputs_embeds,
        }

        # Include 3D scene embeddings for the first generation step
        if mnet_inputs is not None and past_key_values is None:
            model_inputs["mnet_inputs"] = mnet_inputs

        if qformer_inputs is not None:
            model_inputs["qformer_inputs"] = qformer_inputs

        if pixel_values is not None and image_sizes is not None and past_key_values is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_sizes"] = image_sizes

        # Include focus_bbox if provided
        if "focus_bbox" in kwargs:
            model_inputs["focus_bbox"] = kwargs["focus_bbox"]

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
        if isinstance(self.fuyu, PeftModel):
            logger.info("Saving the LoRA.")
            self.fuyu.save_pretrained(save_directory)
        else:
            logger.info("Not saving the frozen Phi3V model.")

        state_dict = self.state_dict()
        all_other_params = {k: v for k, v in state_dict.items() if "llm" not in k}
        logger.info(f"Saving all other params: {all_other_params.keys()}")
        torch.save(all_other_params, os.path.join(save_directory, "other_params.pth"))

    def load_pretrained(self, save_directory):
        logger.info(f"Loading non-LLM checkpoint from {save_directory}")
        all_other_params = torch.load(os.path.join(save_directory, "other_params.pth"))
        message = self.load_state_dict(all_other_params, strict=False)
        logger.info(message)
        del all_other_params
