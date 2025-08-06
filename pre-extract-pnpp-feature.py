import torch
import numpy as np
import argparse
from models.fuyu_3d import PointNetPPTokenizer
from fuyu_utils import PointCloudProcessMixin, VisualInstructionTuningDataset3D
from transformers.configuration_utils import PretrainedConfig
from logging import getLogger
import os
import pickle
from utils.pc_utils import random_sampling
from tqdm.auto import tqdm

logger = getLogger(__name__)
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/scratch/generalvision/mowentao/SQA3D/ScanQA/outputs/2023-06-10_00-11-47_AUXI/model_last.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/scannetv2-pnpp-feature.pkl",
        help="Path to save the extracted features",
    )
    return parser.parse_args()


class ScanNetDataset(torch.utils.data.Dataset, PointCloudProcessMixin):
    @staticmethod
    def get_scene_path() -> str:
        return "/scratch/generalvision/mowentao/ScanQA/data/scannet/scannet_data"

    @staticmethod
    def get_multiview_path() -> str:
        return "/scratch/generalvision/mowentao/ScanQA/data/scannet/scannet_data/enet_feats_maxpool"

    def __len__(self):
        return len(self.scene_list)
    
    def __init__(self, num_points: int = 40_000):
        self.scans_path = self.get_scene_path()
        self.num_points = num_points

        self._load()  # load scene data

    def _load(self):
        """
        Load 3D scene, instance and object information
        """
        logger.info("Loading scene (ScanNet) data...")
        # add scannet data
        scene_file = "/scratch/generalvision/mowentao/ScanQA/data/scannet/meta_data/scannetv2.txt"
        scene_file = open(scene_file, "r").readlines()
        self.scene_list = [x.strip() for x in scene_file]

        # load scene data
        self.scene_data = {}

        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(self.scans_path, scene_id) + "_aligned_vert.npy"
            )  # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(
                os.path.join(self.scans_path, scene_id) + "_ins_label.npy"
            )
            self.scene_data[scene_id]["semantic_labels"] = np.load(
                os.path.join(self.scans_path, scene_id) + "_sem_label.npy"
            )
            self.scene_data[scene_id]["instance_bboxes"] = np.load(
                os.path.join(self.scans_path, scene_id) + "_aligned_bbox.npy"
            )

    def __getitem__(self, idx):
        scene_id = self.scene_list[idx]
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]
        # instance_labels = self.scene_data[scene_id]['instance_labels']
        # semantic_labels = self.scene_data[scene_id]['semantic_labels']
        # instance_bboxes = self.scene_data[scene_id]['instance_bboxes']

        # add xyzrgb
        point_cloud = mesh_vertices[:, 0:6]
        point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0

        # add height
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # add multiview
        enet_feats_file = os.path.join(self.get_multiview_path(), scene_id) + ".pkl"
        multiview = pickle.load(open(enet_feats_file, "rb"))
        point_cloud = np.concatenate([point_cloud, multiview], 1)  # p (50000, 135)

        point_cloud, choices = random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        return {"point_cloud": point_cloud, "scene_id": scene_id}


def collate_fn(batch):
    point_clouds = [x["point_cloud"] for x in batch]
    scene_ids = [x["scene_id"] for x in batch]
    return {
        "point_cloud": torch.from_numpy(np.stack(point_clouds, axis=0)).float(),
        "scene_id": scene_ids,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetPPTokenizer(PretrainedConfig(in_channels=132, freeze_pnpp=True))
    model = model.to(device)
    model.load_pretrained(args.model_path)
    model.eval()

    dataset = ScanNetDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    features = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            # batch = batch.to(model.device)
            point_cloud = batch["point_cloud"].to(device)
            object_feature, object_mask = model(point_cloud)

        for i, scene_id in enumerate(batch["scene_id"]):
            features.append(
                {
                    "scene_id": scene_id,
                    "feature": object_feature[i].cpu(),
                    "mask": object_mask[i].cpu(),
                }
            )

    with open(args.output_path, "wb") as f:
        torch.save(features, f)


main()