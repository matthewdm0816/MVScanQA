import torch
import torch.nn as nn
import transformers
from transformers import BertConfig, BertModel, BertTokenizer, BertPreTrainedModel

# a simple Point Cloud Encoder
# of PointMixer architecture: group points randomly, with extensive groups. #=> which means each point is covered by multiple groups
# class PointMixerLayer(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.num_groups = config.num_groups
#         self.num_points = config.num_points
#         self.hidden_size = config.hidden_size
#         self.dropout = nn.Dropout(config.dropout)
#         self.attention = nn.MultiheadAttention(config.hidden_size, config.num_heads, config.dropout)
#         self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm1 = nn.LayerNorm(config.hidden_size)
#         self.layer_norm2 = nn.LayerNorm(config.hidden_size)
#         self.activation = nn.GELU()

#     def forward(self, input_point_embeds, attention_mask=None):
#         # input_point_embeds: (batch_size, num_points, hidden_size)
#         # attention_mask: (batch_size, num_points)
#         # output: (batch_size, N_groups, hidden_size)

#         # group points randomly
#         batch_size, num_points, hidden_size = input_point_embeds.size()
#         assert num_points == self.num_points
#         assert hidden_size == self.hidden_size
#         input_point_embeds = input_point_embeds.view(batch_size, self.num_groups, self.num_points // self.num_groups, self.hidden_size)


# class PointMixerEncoder(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.layers = nn.ModuleList([PointMixerLayer(config) for _ in range(config.num_hidden_layers)])


class PointMixerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.embed_size, config.hidden_size)
        # self.encoder = PointMixerEncoder(config)
        self.encoder = BertModel(config) # a simple Transformer Encoder
        # TODO: add decoder for MAE-like training

    def forward(self, input_point_embeds, attention_mask=None):
        # input_point_embeds: (batch_size, num_points, embed_size)
        # attention_mask: (batch_size, num_points)
        # output: (batch_size, hidden_size)
        input_point_embeds = self.linear(input_point_embeds) # (batch_size, num_points, hidden_size)

        # group points randomly
        batch_size, num_points, hidden_size = input_point_embeds.size()
        assert num_points == self.config.num_points
        assert hidden_size == self.config.hidden_size

        group_indices = torch.randperm(num_points).view(batch_size, self.config.num_groups, -1).to(input_point_embeds.device)

        input_group_embeds = torch.gather(input_point_embeds, 1, group_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size)).squeeze(-2)

        return output

        