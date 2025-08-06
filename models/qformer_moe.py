import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from transformers.models.instructblip.modeling_instructblip import (
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig,
    InstructBlipPreTrainedModel,
    InstructBlipQFormerEmbeddings,
    InstructBlipQFormerEncoder,
    InstructBlipQFormerLayer,
    InstructBlipQFormerAttention,
    InstructBlipQFormerIntermediate,
    InstructBlipQFormerOutput,
)

from transformers.activations import ACT2FN

from transformers.configuration_utils import PretrainedConfig

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
)

from transformers.pytorch_utils import apply_chunking_to_forward

from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

# MIXTRAL_ATTENTION_CLASSES = {
#     "eager": MixtralAttention,
#     "flash_attention_2": MixtralFlashAttention2,
#     "sdpa": MixtralSdpaAttention,
# }


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = F.gelu

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
    

# class InstructBlipQFormerIntermediateBlockSparse(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states


# # Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->InstructBlipQFormer
# class InstructBlipQFormerOutput(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#     def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + input_tensor)
#         return hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config, expert_module, hidden_dim: int, output_dim: int):
        super().__init__()
        # self.hidden_dim = config.hidden_size
        # self.ffn_dim = config.intermediate_size
        # self.num_experts = config.num_local_experts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        self.experts = nn.ModuleList([expert_module(config) for _ in range(self.num_experts)])

        # Jitter parameters
        # self.jitter_noise = config.router_jitter_noise
        # self.jitter_noise = config.get("router_jitter_noise", 0.0)
        self.jitter_noise = getattr(config, "router_jitter_noise", 0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        hidden_states = hidden_states.contiguous()
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) # [bs * num_token, num_experts]
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.output_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.output_dim)
        return final_hidden_states, router_logits


class InstructBlipQFormerEncoderWithMoEConfig(InstructBlipQFormerConfig):
    def __init__(self, num_experts=8, num_experts_per_tok=2, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

class InstructBlipQFormerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense_intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act
        
        # self.dense_output = nn.Linear(config.intermediate_size, config.hidden_size)
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.intermediate = InstructBlipQFormerIntermediate(config)
        self.output = InstructBlipQFormerOutputNonResidual(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.dense_intermediate(hidden_states)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        # hidden_states = self.intermediate_act_fn(self.dense_intermediate(hidden_states))

        # hidden_states = self.dense_output(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        # hidden_states = self.LayerNorm(self.dropout(self.dense_output(hidden_states)))
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states)

        return hidden_states
        
class InstructBlipQFormerOutputNonResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class InstructBlipQFormerLayerWithMoE(nn.Module):
    def __init__(self, config, layer_idx):
        # super().__init__(config, layer_idx=layer_idx)
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = InstructBlipQFormerAttention(config)

        self.layer_idx = layer_idx

        if layer_idx % config.cross_attention_frequency == 0:
            self.crossattention = InstructBlipQFormerAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False

        # self.intermediate = InstructBlipQFormerIntermediate(config)
        # self.output = InstructBlipQFormerOutput(config)
        # here hidden_dim is the input size of the gate layers (and is the input size of the experts)
        # self.intermediate = MixtralSparseMoeBlock(config, InstructBlipQFormerIntermediate, hidden_dim=config.hidden_size, output_dim=config.intermediate_size)
        # self.output = MixtralSparseMoeBlock(config, InstructBlipQFormerOutputNonResidual, hidden_dim=config.intermediate_size, output_dim=config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = MixtralSparseMoeBlock(config, InstructBlipQFormerMLP, hidden_dim=config.hidden_size, output_dim=config.hidden_size)

        self.intermediate_query = InstructBlipQFormerIntermediate(config)
        self.output_query = InstructBlipQFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                # add cross attentions if we output attention weights
                outputs = outputs + cross_attention_outputs[1:-1]

            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            if attention_output.shape[1] > query_length:
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # intermediate_output, router_logits_intermediate = self.intermediate(attention_output)
        # layer_output, router_logits_output = self.output(intermediate_output)
        layer_output, router_logits = self.ffn(attention_output)
        layer_output = self.LayerNorm(layer_output + attention_output) # add back the residual connection
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


def build_moe_qformer_from_config(qformer_name=None, config=None, num_experts=8, num_experts_per_tok=2, init_with_same_pretrained_weights=True):
    config.num_experts = num_experts
    config.num_experts_per_tok = num_experts_per_tok

    if qformer_name is None:
        logger.info(f"Building new moe-qformer with config: {config}")
        model = InstructBlipQFormerModel(config)
    else:
        logger.info(f"Loading pretrained moe-qformer from {qformer_name}")
        model = InstructBlipQFormerModel.from_pretrained(qformer_name, config=config)
    
    # init a new model, which each FFN (intermediate and output) replaced by MoE
    #   and we init each expert with the same weights from the original model

    # replace the FFN in each layer with MoE
    for i, layer in enumerate(model.encoder.layer):
        old_layer = model.encoder.layer[i]

        new_layer = InstructBlipQFormerLayerWithMoE(config, i)

        # init the MoE layer with the same weights as the original layer, if it exists
        if qformer_name is not None and init_with_same_pretrained_weights:
            msg = new_layer.load_state_dict(old_layer.state_dict(), strict=False) # this will skip the MoE layers

            logger.info(f"[MoE Init] Layer {i}: {msg}")

            # copy the weights of the intermediate and output layers to the MoE layers
            if hasattr(new_layer, "intermediate"):
                for expert in new_layer.intermediate.experts:
                    msg = expert.load_state_dict(old_layer.intermediate.state_dict(), strict=False)
                    logger.info(f"[MoE Init] Layer Intermediate {i}: {msg}")

            if hasattr(new_layer, "output"):
                for expert in new_layer.output.experts:
                    msg = expert.load_state_dict(old_layer.output.state_dict(), strict=False)
                    logger.info(f"[MoE Init] Layer Output {i}: {msg}")

            if hasattr(new_layer, "ffn"):
                for expert in new_layer.ffn.experts:
                    msg = expert.intermediate.load_state_dict(old_layer.intermediate.state_dict(), strict=False)
                    logger.info(f"[MoE Init] Layer -FFN- Intermediate {i}: {msg}")
                    msg = expert.output.load_state_dict(old_layer.output.state_dict(), strict=False)
                    logger.info(f"[MoE Init] Layer -FFN- Output {i}: {msg}")

        model.encoder.layer[i] = new_layer

    return model



if __name__ == "__main__":
    pass
