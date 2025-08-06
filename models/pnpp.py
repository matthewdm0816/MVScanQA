"""
Separate PointNet++ model
"""

from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule


import torch
import torch.nn as nn

from logging import getLogger

logger = getLogger(__name__)

class PointNetPP(nn.Module):
    def __init__(
        self, 
        # proposal
        num_object_class,
        input_feature_dim,
        num_heading_bin,
        num_size_cluster,
        mean_size_arr,
        num_proposal=256,
        vote_factor=1,
        sampling="vote_fps",
        seed_feat_dim=256,
        proposal_size=128,
        pointnet_width=1,
        pointnet_depth=2,
        vote_radius=0.3,
        vote_nsample=16,
        **kwargs,
    ):
        super().__init__()

        self.detection_backbone = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=pointnet_width,
            depth=pointnet_depth,
            seed_feat_dim=seed_feat_dim,
        )
        # Hough voting
        self.voting_net = VotingModule(vote_factor, seed_feat_dim)

        # Vote aggregation and object proposal
        self.proposal_net = ProposalModule(
            num_object_class,
            num_heading_bin,
            num_size_cluster,
            mean_size_arr,
            num_proposal,
            sampling,
            seed_feat_dim=seed_feat_dim,
            proposal_size=proposal_size,
            radius=vote_radius,
            nsample=vote_nsample,
        )

    def forward(self, point_cloud):
        data_dict = {
            "point_clouds": point_cloud,
        }
        data_dict = self.detection_backbone(data_dict)

        xyz = data_dict["fp2_xyz"]
        features = data_dict[
            "fp2_features"
        ]  # batch_size, seed_feature_dim, num_seed, (16, 256, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz

        data_dict["seed_features"] = features
        xyz, features = self.voting_net(
            xyz, features
        )  # batch_size, vote_feature_dim, num_seed * vote_factor, (16, 256, 1024)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        data_dict = self.proposal_net(xyz, features, data_dict)

        # unpack outputs from detection branch
        object_feat = data_dict["aggregated_vote_features"]

        # same meaning as in transformers lib (no need to invert)
        object_mask = data_dict["bbox_mask"].bool().detach()

        return object_feat, object_mask
    
    def load_pretrained(self, ckpt_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(ckpt_path, map_location="cpu")
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(pretrained_dict, "module.")

        # 1. filter out unnecessary keys
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(list(pretrained_dict.keys()))
        # # 2. overwrite entries in the existing state dict
        # model_dict.update(pretrained_dict)
        # # 3. load the new state dict
        # self.load_state_dict(model_dict)

        missing, unexpected = self.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"Missing keys: {missing}")
        logger.info(f"Unexpected keys: {unexpected}")