'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on huggingface code base
 * https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/models/bert
'''

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig
from icecream import ic
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        self.config = config

    def forward(
        self, input_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = False   
            
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)         

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class BertOutputParallel(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        self.LayerNorms = nn.ModuleList([nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(1)])
    
    def forward(self, hidden_states, input_tensor, layernorm_idx=0):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_layer = self.LayerNorm if layernorm_idx == 0 else self.LayerNorms[layernorm_idx - 1]
        hidden_states = ln_layer(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)      
        self.layer_num = layer_num          
        if self.config.add_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=self.config.add_cross_attention)
        self.intermediate = BertIntermediate(config)
        # self.output = BertOutput(config)
        self.output = BertOutputParallel(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        mode=None,
        layernorm_idx=0,
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

        outputs = self_attention_outputs[1:-1] # basically the attention weights
        present_key_value = self_attention_outputs[-1]

        if mode=='multimodal':
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights                               
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output, layernorm_idx
        )
        outputs = (layer_output,) + outputs 

        outputs = outputs + (present_key_value,) # feature-out, attention-out, cross-attention-out, present_key_value

        return outputs

    def feed_forward_chunk(self, attention_output, layernorm_idx):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, layernorm_idx=layernorm_idx)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config,i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multimodal',
        layernorm_idx=0,
        forward_layers=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        if forward_layers is None:
            forward_layers = range(self.config.num_hidden_layers)
        else:
            forward_layers = [i for i in range(self.config.num_hidden_layers) if i in forward_layers]
        # for i in range(self.config.num_hidden_layers):
        for i in forward_layers:
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    mode=mode,
                    layernorm_idx=layernorm_idx,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    mode=mode,
                    layernorm_idx=layernorm_idx,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
class BertEncoderTwin(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.num_hidden_layers_twin = getattr(config, 'num_hidden_layers_twin', config.num_hidden_layers)
        self.layer_twin = nn.ModuleList([BertLayer(config,i) for i in range(self.num_hidden_layers_twin)])

    def init_twin(self):
        logger.info('Initializing twin encoder')
        # self.layer_twin.load_state_dict(self.layer.state_dict())
        for i in range(self.num_hidden_layers_twin):
            self.layer_twin[i].load_state_dict(self.layer[i].state_dict())

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_twin=None,
        encoder_attention_mask_twin=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multimodal',
        layernorm_idx=0,
        forward_layers=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        if forward_layers is None:
            forward_layers = range(self.config.num_hidden_layers)
        else:
            forward_layers = [i for i in range(self.config.num_hidden_layers) if i in forward_layers]
        # for i in range(self.config.num_hidden_layers):
        hidden_states_twin = hidden_states.clone()
        for i in forward_layers:
            layer_module = self.layer[i]
            layer_module_twin = self.layer_twin[i] if i < self.num_hidden_layers_twin else None
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            encoder_hidden_states_mix = torch.cat((encoder_hidden_states, hidden_states_twin), dim=1)
            # encoder_attention_mask = torch.cat((encoder_attention_mask, attention_mask), dim=1)

            encoder_hidden_states_twin_mix = torch.cat((encoder_hidden_states_twin, hidden_states), dim=1)
            # encoder_attention_mask_twin = torch.cat((encoder_attention_mask_twin, attention_mask), dim=1)

            # ic(i, encoder_hidden_states.shape, encoder_attention_mask.shape)
            # ic(i, encoder_hidden_states_twin.shape, encoder_attention_mask_twin.shape)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states_mix,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                mode=mode,
                layernorm_idx=layernorm_idx,
            )
            if output_attentions:
                # print(f"Layer {i} 2D")
                # ic(len(layer_outputs))
                self_attentions = (layer_outputs[1],) 
                # print(f"2D self attention shape: {self_attentions[0].shape}") # shall be [B, N_head, L, L_2D]
                if all_cross_attentions is not None:
                    cross_attentions = (layer_outputs[-2],)
                    # print(f"2D cross attention shape: {cross_attentions[0].shape}")
                    
            if layer_module_twin is not None:
                layer_outputs_twin = layer_module_twin(
                    hidden_states_twin,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states_twin_mix,
                    encoder_attention_mask_twin,
                    past_key_value,
                    output_attentions,
                    mode=mode,
                    layernorm_idx=layernorm_idx,
                )
                hidden_states_twin = layer_outputs_twin[0]
                # NOTE: how to record cross attentions?
                if output_attentions:
                    # print(f"Layer {i} 3D")
                    # ic(len(layer_outputs))
                    self_attentions_twin = layer_outputs_twin[1]
                    self_attentions = self_attentions + (self_attentions_twin, )
                    # print(f"3D self attention shape: {self_attentions_twin.shape}") # shall be [B, N_head, L, L]

                    if all_cross_attentions is not None:
                        cross_attentions_twin = layer_outputs_twin[-2]
                        cross_attentions = cross_attentions + (cross_attentions_twin, )
                        # print(f"3D cross attention shape: {cross_attentions_twin.shape}") # shall be [B, N_head, L, L_3D]

            hidden_states = layer_outputs[0]
            # hidden_states_twin = layer_outputs_twin[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (self_attentions,) # latest layer's self attention put at the end
                if all_cross_attentions is not None:
                    all_cross_attentions = all_cross_attentions + (cross_attentions,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, hidden_states_twin)


        if not return_dict:
            return tuple(
                v
                for v in [
                    (hidden_states, hidden_states_twin),
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=(hidden_states, hidden_states_twin),
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()
 

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device, is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape

                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
   
                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )                     

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multimodal',
        layernorm_idx=0,
        forward_layers=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, 
                                                                                 device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
            layernorm_idx=layernorm_idx,
            forward_layers=forward_layers,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    
# class BertSparseCrossAttentionMoEBlock(nn.Module):
class BertSparseCrossAttentionMoELayer(BertLayer):
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

    def __init__(self, config, layer_num, expert_module=None):
        # super().__init__()
        super().__init__(config, layer_num)

        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        self.hidden_dim = config.hidden_size
        self.output_dim = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.biased_indices = getattr(config, "biased_indices", [0])
        self.expert_bias = torch.zeros(self.num_experts)
        self.expert_bias[self.biased_indices] = 1.0
        # self.expert_bias = torch.Parameter(self.expert_bias, requires_grad=False)
        self.expert_bias = nn.Parameter(self.expert_bias, requires_grad=False)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        # self.experts = nn.ModuleList([expert_module(config) for _ in range(self.num_experts)]) 
        assert self.config.add_cross_attention, "This layer must have cross-attention"
        self.cross_attention_experts = nn.ModuleList([
            BertAttention(config, is_cross_attention=self.config.add_cross_attention) for _ in range(self.num_experts)
        ])
        # experts are the BertLayer(s) that require self and cross attention

        # Jitter parameters
        self.jitter_noise = getattr(config, "router_jitter_noise", 0.0)

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor,
                common_encoder_hidden_states: torch.Tensor, 
                common_encoder_hidden_states_mask: torch.Tensor,
                encoder_hidden_states: torch.Tensor, 
                encoder_hidden_states_mask: torch.Tensor,
                # *args,
                head_mask=None,
                past_key_value=None,
                output_attentions=False,
                mode=None,
                layernorm_idx=0,
            ) -> torch.Tensor:
        # Self attention keeps the same

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

        outputs = self_attention_outputs[1:-1] # basically the attention weights
        present_key_value = self_attention_outputs[-1]


        """
        cross-attention with view-based routing and block-sparse MoE
        """
        hidden_states = attention_output 
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        _, num_encoder_hidden_states, encoder_sequence_length, encoder_hidden_dim = encoder_hidden_states.shape
        _, common_sequence_length, common_hidden_dim = common_encoder_hidden_states.shape
        assert num_encoder_hidden_states == self.num_experts
        assert common_hidden_dim == encoder_hidden_dim

        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        hidden_states = hidden_states.contiguous()
        hidden_states = hidden_states.view(-1, hidden_dim) # [bs * num_token, hidden_dim]
        router_logits = self.gate(hidden_states) # (batch * sequence_length, n_experts)
        router_logits += self.expert_bias[None, :] # add bias to some experts

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [bs * num_token, num_experts], [bs * num_token, top_k]
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True) 

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype).view(batch_size, sequence_length, -1) # [bs, num_token, num_experts]

        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, self.output_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # [bs * num_token, top_k, num_experts] 
        expert_mask = expert_mask.view(batch_size, sequence_length, *expert_mask.shape[1:])
        # expert_mask ~ [bs, num_token, top_k, num_experts]

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # expert_layer: Union[BertLayer, nn.Module] = self.experts[expert_idx]
            expert_layer: BertAttention = self.cross_attention_experts[expert_idx]
            # idx, top_x = torch.where(expert_mask[expert_idx]) 

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            # prepare padding indices
            this_expert_mask = expert_mask[..., expert_idx].any(dim=-1) # [bs, num_token]
            this_expert_indices = this_expert_mask.cumsum(dim=-1) # [bs, num_token], the index to the padded states (to be scattered)
            this_expert_indices[~this_expert_mask] = 0 # for tokens that are not in the top-k, set to 0, and later 0-th token will be removed
            this_expert_max_length = this_expert_mask.sum(dim=-1).max().item()

            logger.info(f"this_expert_max_length: {this_expert_max_length}")
            logger.info(f"attention_mask: {attention_mask.shape}, min: {attention_mask.min()}, max: {attention_mask.max()}")


            # prepare the padded hidden states
            # current_state ~ [bs, max_num_token, hidden_dim]
            current_state = torch.zeros(batch_size, this_expert_max_length + 1, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            neg_val = attention_mask.min()
            if neg_val >= 0:
                neg_val = torch.finfo(hidden_states.dtype).min
            current_state_mask = torch.zeros(batch_size, this_expert_max_length + 1, device=hidden_states.device, dtype=hidden_states.dtype) + neg_val 
            # already inverted, 0 is non-masked, -inf is masked

            # logger.info()

            current_state.scatter_add_(1, this_expert_indices[:, :, None].expand(-1, -1, hidden_dim), hidden_states.view(batch_size, sequence_length, hidden_dim))
            current_state_mask.scatter_(1, this_expert_indices, attention_mask.squeeze())[:, None, None, :] # [bs, 1, 1, max_num_token]
            current_state = current_state[:, 1:] # remove the 0-th token
            current_state_mask = current_state_mask[:, 1:] # remove the 0-th token
            
            # current_state_mask ~ [bs, max_num_token]

            logger.info(f"current_state: {current_state.shape}")
            logger.info(f"current_state_mask: {current_state_mask.shape}")

            logger.info(f"common_encoder_hidden_states: {common_encoder_hidden_states.shape}")
            logger.info(f"encoder_hidden_states: {encoder_hidden_states.shape}")
            logger.info(f"common_encoder_hidden_states_mask: {common_encoder_hidden_states_mask.shape}")
            logger.info(f"encoder_hidden_states_mask: {encoder_hidden_states_mask.shape}")

            expert_cross_attention_hidden_states = torch.cat([common_encoder_hidden_states, encoder_hidden_states[:, expert_idx, :, :]], dim=1)
            expert_cross_attention_hidden_states_mask = torch.cat([common_encoder_hidden_states_mask, encoder_hidden_states_mask], dim=-1)

            cross_attention_outputs = expert_layer(
                current_state,
                # attention_mask=attention_mask,
                attention_mask=current_state_mask,
                encoder_hidden_states=expert_cross_attention_hidden_states,
                encoder_attention_mask=expert_cross_attention_hidden_states_mask,
                # *args,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            # ) * routing_weights[top_x, idx, None]
            # ) * routing_weights[:, :, expert_idx][:, :, None]
            current_hidden_states = cross_attention_outputs[0]
            if expert_idx == 0:
                outputs = outputs + cross_attention_outputs[1:-1]

            # gather the hidden states back to the original shape
            # output_states = torch.
            # FIXME: this is not the right way to gather the hidden states
            current_hidden_states = torch.gather(current_hidden_states, 1, this_expert_indices[:, :, None].expand(-1, -1, self.output_dim))
            # current_hidden_states ~ [bs, num_token, hidden_dim]

            # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states += current_hidden_states * routing_weights[:, :, expert_idx][:, :, None]

        final_hidden_states = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, final_hidden_states, layernorm_idx
        )
        outputs = (final_hidden_states,) + outputs + (present_key_value,)

        return final_hidden_states, router_logits

class BertSparseCrossAttentionViewMoELayer(BertLayer):
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

    def __init__(self, config, layer_num, expert_module=None):
        # super().__init__()
        super().__init__(config, layer_num)

        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        self.hidden_dim = config.hidden_size
        self.output_dim = config.hidden_size

        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.biased_indices = getattr(config, "biased_indices", [0])
        self.expert_bias = torch.zeros(self.num_experts)
        self.expert_bias[self.biased_indices] = 1.0
        # self.expert_bias = torch.Parameter(self.expert_bias, requires_grad=False)
        self.expert_bias = nn.Parameter(self.expert_bias, requires_grad=False)

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        # self.experts = nn.ModuleList([expert_module(config) for _ in range(self.num_experts)]) 
        assert self.config.add_cross_attention, "This layer must have cross-attention"
        self.cross_attention_experts = nn.ModuleList([
            BertAttention(config, is_cross_attention=self.config.add_cross_attention) for _ in range(self.num_experts)
        ])
        # experts are the BertLayer(s) that require self and cross attention

        # Jitter parameters
        self.jitter_noise = getattr(config, "router_jitter_noise", 0.0)

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.Tensor,
                common_encoder_hidden_states: torch.Tensor, 
                common_encoder_hidden_states_mask: torch.Tensor,
                encoder_hidden_states: torch.Tensor, 
                encoder_hidden_states_mask: torch.Tensor,
                # *args,
                head_mask=None,
                past_key_value=None,
                output_attentions=False,
                mode=None,
                layernorm_idx=0,
            ) -> torch.Tensor:
        # Self attention keeps the same

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

        outputs = self_attention_outputs[1:-1] # basically the attention weights
        present_key_value = self_attention_outputs[-1]


        """
        cross-attention with view-based routing and block-sparse MoE
        sample-wise routing
        """
        hidden_states = attention_output 
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        _, num_encoder_hidden_states, encoder_sequence_length, encoder_hidden_dim = encoder_hidden_states.shape
        _, common_sequence_length, common_hidden_dim = common_encoder_hidden_states.shape
        assert num_encoder_hidden_states == self.num_experts
        assert common_hidden_dim == encoder_hidden_dim

        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        hidden_states = hidden_states.contiguous()
        pooled_hidden_states = hidden_states.mean(dim=1) # [bs, hidden_dim]

        router_logits = self.gate(pooled_hidden_states) # [bs, num_experts]
        router_logits += self.expert_bias[None, :] # add bias to some experts

        # hard top-k routing
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float) # [bs, num_experts]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1) # [bs, top_k], [bs, top_k]
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True).to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size, sequence_length, self.output_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        # expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # [bs * num_token, top_k, num_experts] 
        # expert_mask = expert_mask.view(batch_size, sequence_length, *expert_mask.shape[1:])

        # expert_mask ~ [bs, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts) # [bs, top_k, num_experts]

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer: BertAttention = self.cross_attention_experts[expert_idx]

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)

            batch_idx, top_x = torch.where(expert_mask[:, :, expert_idx]) # [num_samples_use_this_expert], [idx_which_expert]
            # logger.info(f"expert {expert_idx} feedforwards {len(batch_idx)} samples")
            # logger.info(f"batch_idx: {batch_idx}")
            if batch_idx.shape[0] == 0:
                continue
            
            current_hidden_states = hidden_states[batch_idx, :, :] # [num_samples_use_this_expert, L, hidden_dim]
            current_state_mask = attention_mask[batch_idx, :] # [num_samples_use_this_expert, L]
            current_routing_weights = routing_weights[batch_idx, top_x] # [num_samples_use_this_expert]

            # logger.info(f"routing_weights: {routing_weights}")

            # logger.info(f"current_hidden_states before: {current_hidden_states.shape}")
            # logger.info(f"current_hidden_states_mask before: {current_state_mask.shape}")
            # logger.info(f"current_routing_weights: {current_routing_weights.shape}")
            # logger.info(f"batch_idx: {batch_idx}, top_x: {top_x}")

            # logger.info(f"common_encoder_hidden_states: {common_encoder_hidden_states.shape}")
            # logger.info(f"encoder_hidden_states: {encoder_hidden_states.shape}")
            # logger.info(f"common_encoder_hidden_states_mask: {common_encoder_hidden_states_mask.shape}")
            # logger.info(f"encoder_hidden_states_mask: {encoder_hidden_states_mask.shape}")

            # expert_cross_attention_hidden_states = torch.cat([common_encoder_hidden_states[batch_idx], encoder_hidden_states[:, expert_idx, :, :][batch_idx]], dim=1)
            # expert_cross_attention_hidden_states_mask = torch.cat([common_encoder_hidden_states_mask[batch_idx], encoder_hidden_states_mask[batch_idx]], dim=-1)

            expert_cross_attention_hidden_states = torch.cat([encoder_hidden_states[:, expert_idx, :, :][batch_idx], common_encoder_hidden_states[batch_idx]], dim=1)
            expert_cross_attention_hidden_states_mask = torch.cat([encoder_hidden_states_mask[batch_idx], common_encoder_hidden_states_mask[batch_idx]], dim=-1)

            # logger.info(f"expert_cross_attention_hidden_states: {expert_cross_attention_hidden_states.shape}")
            # logger.info(f"expert_cross_attention_hidden_states_mask: {expert_cross_attention_hidden_states_mask.shape}")

            cross_attention_outputs = expert_layer(
                current_hidden_states,
                attention_mask=current_state_mask,
                encoder_hidden_states=expert_cross_attention_hidden_states,
                encoder_attention_mask=expert_cross_attention_hidden_states_mask,
                # *args,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )
            current_hidden_states = cross_attention_outputs[0]
            if expert_idx == 0:
                outputs = outputs + cross_attention_outputs[1:-1]

            # gather the hidden states back to the original shape
            current_hidden_states = current_hidden_states * current_routing_weights[:, None, None] 
            # ~ [num_samples_use_this_expert, L, hidden_dim]
            # logger.info(f"current_hidden_states after: {current_hidden_states.shape}")
            # logger.info(f"batch_idx: {batch_idx.shape}, top_x: {top_x.shape}")

            final_hidden_states.index_add_(0, batch_idx, current_hidden_states.to(hidden_states.dtype))
            

        final_hidden_states = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, final_hidden_states, layernorm_idx
        )
        outputs = (final_hidden_states,) + outputs + (present_key_value,)

        # return final_hidden_states, router_logits
        return outputs


class BertEncoderCrossAttentionMoE(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        # self.num_hidden_layers_twin = getattr(config, 'num_hidden_layers_twin', config.num_hidden_layers)
        # self.layer_twin = nn.ModuleList([BertLayer(config,i) for i in range(self.num_hidden_layers_twin)])
        # self.layer_twin = nn.ModuleList([BertSparseCrossAttentionMoEBlock(config, lambda config: BertLayer(config, i), config.hidden_size, config.hidden_size) for i in range(config.num_hidden_layers)])
        self.config.moe_mode = getattr(config, 'moe_mode', 'view-wise')
        if self.config.moe_mode == 'view-wise':
            self.layer_twin = nn.ModuleList([BertSparseCrossAttentionViewMoELayer(config, i) for i in range(config.num_hidden_layers)])
        elif self.config.moe_mode == 'token-wise':
            self.layer_twin = nn.ModuleList([BertSparseCrossAttentionMoELayer(config, i) for i in range(config.num_hidden_layers)])


    def init_twin(self):
        logger.info('Initializing twin expert encoders')
        for i in range(self.config.num_hidden_layers):
            msg = self.layer_twin[i].load_state_dict(self.layer[i].state_dict(), strict=False) # NOTE: there are additional parameters in the expert, so strict=False
            logger.info(f"Layer {i} twin initialized with {msg}")
            # for expert in self.layer_twin[i].experts:
            #     expert.load_state_dict(self.layer[i].state_dict()) # initialize the expert with the same weights as the main model
            expert_msgs = []
            for expert in self.layer_twin[i].cross_attention_experts:
                msg = expert.load_state_dict(self.layer[i].crossattention.state_dict())
                expert_msgs.append(repr(msg))
            logger.info(f"Layer {i} experts initialized with [{', '.join(expert_msgs)}]")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None, # 2d
        encoder_attention_mask=None,
        encoder_hidden_states_twin=None, # 3d
        encoder_attention_mask_twin=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multimodal',
        layernorm_idx=0,
        forward_layers=None,
    ):
        # reverse inputs
        # for code compatiblity, it is reversed as in non-moe model
        # then, encoder_hidden_states is 3d that goes no view-MoE, 
        # encoder_hidden_states_twin is 2d that goes to view-MoE (multi-view provided for each expert)
        encoder_hidden_states, encoder_hidden_states_twin = encoder_hidden_states_twin, encoder_hidden_states
        encoder_attention_mask, encoder_attention_mask_twin = encoder_attention_mask_twin, encoder_attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        
        if forward_layers is None:
            forward_layers = range(self.config.num_hidden_layers)
        else:
            forward_layers = [i for i in range(self.config.num_hidden_layers) if i in forward_layers]
        hidden_states_twin = hidden_states.clone()
        
        for i in forward_layers:
            layer_module = self.layer[i]
            layer_module_twin: BertSparseCrossAttentionMoELayer = self.layer_twin[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            encoder_hidden_states_mix = torch.cat((encoder_hidden_states, hidden_states_twin), dim=1)
            encoder_attention_mask_mix = torch.cat((encoder_attention_mask, attention_mask), dim=-1)

            # encoder_hidden_states_twin_mix = torch.cat((encoder_hidden_states_twin, hidden_states), dim=1)
            # common_encoder_hidden_states_twin = hidden_states
            # common_encoder_attention_mask_twin = attention_mask

            # --- forward main branch ---

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states_mix,
                # encoder_attention_mask,
                encoder_attention_mask_mix,
                past_key_value,
                output_attentions,
                mode=mode,
                layernorm_idx=layernorm_idx,
            )
            # --- forward twin branch ---

            layer_outputs_twin = layer_module_twin(
                hidden_states_twin,
                attention_mask=attention_mask,
                common_encoder_hidden_states=hidden_states, # other branch intermediate hidden states
                common_encoder_hidden_states_mask=attention_mask, # other branch input mask
                encoder_hidden_states=encoder_hidden_states_twin, # multi-view expert input
                encoder_hidden_states_mask=encoder_attention_mask_twin, # multi-view expert mask, actually non-multi. same for each view
                head_mask=layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                mode=mode,
                layernorm_idx=layernorm_idx,
            )

            # --- update hidden states ---
            hidden_states = layer_outputs[0]
            hidden_states_twin = layer_outputs_twin[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, hidden_states_twin)


        if not return_dict:
            return tuple(
                v
                for v in [
                    (hidden_states, hidden_states_twin),
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=(hidden_states, hidden_states_twin),
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
class BertModelTwin(BertModel):
    def __init__(self, config, add_pooling_layer=True, twin_mode='single'):
        super().__init__(config)
        self.config = config

        if twin_mode == 'single':
            self.encoder = BertEncoderTwin(config)
        elif twin_mode == 'moe':
            self.encoder = BertEncoderCrossAttentionMoE(config)
            # TODO: swap twin and main for MoE mode.
        else:
            raise ValueError(f"twin_mode {twin_mode} is not supported")

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.pooler_twin = BertPooler(config) if add_pooling_layer else None
        self.twin_mode = twin_mode

    def init_twin(self):
        self.encoder.init_twin()
        if self.pooler is not None:
            self.pooler_twin.load_state_dict(self.pooler.state_dict())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_hidden_states_twin=None,
        encoder_attention_mask_twin=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multimodal',
        layernorm_idx=0,
        forward_layers=None,
        
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
            
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, 
                                                                                 device, is_decoder)

        if self.twin_mode == 'single':
            # pre-concat masks
            encoder_attention_mask = torch.cat((encoder_attention_mask, attention_mask), dim=1)
            encoder_attention_mask_twin = torch.cat((encoder_attention_mask_twin, attention_mask), dim=1)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            elif len(encoder_hidden_states.size()) == 4:
                # multi-view hidden_states of shape [B, num_view, L, H]
                encoder_batch_size, num_views, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, num_views, encoder_sequence_length)
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if encoder_hidden_states_twin is not None:
            if type(encoder_hidden_states_twin) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states_twin[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states_twin.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask_twin) == list:
                encoder_extended_attention_mask_twin = [self.invert_attention_mask(mask) for mask in encoder_attention_mask_twin]
            elif encoder_attention_mask_twin is None:
                encoder_attention_mask_twin = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask_twin = self.invert_attention_mask(encoder_attention_mask_twin)
            else:    
                encoder_extended_attention_mask_twin = self.invert_attention_mask(encoder_attention_mask_twin)
        else:
            encoder_extended_attention_mask_twin = None

        # extended_attention_mask(s): [B, 1, 1, L]

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_hidden_states_twin=encoder_hidden_states_twin,
            encoder_attention_mask_twin=encoder_extended_attention_mask_twin,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
            layernorm_idx=layernorm_idx,
            forward_layers=forward_layers,
        )
        sequence_output, sequence_output_twin = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        pooled_output_twin = self.pooler(sequence_output_twin) if self.pooler is not None else None

        if not return_dict:
            return ((sequence_output, sequence_output_twin), pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=(sequence_output, sequence_output_twin),
            pooler_output=(pooled_output, pooled_output_twin),
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
    

class BertPrefixModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multimodal',
        prefix_embeds=None,
        prefix_attention_mask=None,
        layernorm_idx=0,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:    
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape 
            device = encoder_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or encoder_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # prepend prefix attention mask
        if prefix_embeds is not None:
            assert prefix_attention_mask is not None
            attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
            seq_length += prefix_attention_mask.shape[1]
            input_shape = (batch_size, seq_length)
            
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, 
                                                                                 device, is_decoder)
        

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            
            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [self.invert_attention_mask(mask) for mask in encoder_attention_mask]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:    
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        # ic(prefix_embeds.shape, prefix_attention_mask.shape, attention_mask.shape)
        # ic(embedding_output.shape)
        # prepend prefix embeds
        if prefix_embeds is not None:
            embedding_output = torch.cat([prefix_embeds, embedding_output], dim=1)

        # ic(embedding_output.shape, extended_attention_mask.shape, encoder_extended_attention_mask.shape)
        
            
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode=mode,
            layernorm_idx=layernorm_idx,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,            
        is_decoder=True,
        reduction='mean',
        mode='multimodal', 
        layernorm_idx=0,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
            layernorm_idx=layernorm_idx,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()  

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1) 
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if reduction=='none':
                lm_loss = lm_loss.view(prediction_scores.size(0),-1).sum(1)               

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
    
class BertLMClassificationModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_logits=False,            
        is_decoder=True,
        reduction='mean',
        mode='multimodal', 
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output[:, :-1, :])
        
        return prediction_scores.contiguous(), outputs.last_hidden_state.contiguous()


    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "past_key_values": past,
            "encoder_hidden_states": model_kwargs.get("encoder_hidden_states", None),
            "encoder_attention_mask": model_kwargs.get("encoder_attention_mask", None),
            "is_decoder": True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past