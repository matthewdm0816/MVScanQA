import torch
import numpy as np
import json
import os
import pickle
import argparse
from tqdm.auto import tqdm
import logging
from utils.pc_utils import random_sampling
# from lib.dataset import DC
from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()
from transformers.modeling_utils import PretrainedConfig
from collections import defaultdict
import torch.nn as nn

# from models.fuyu_3d import Vote2CapDETRTokenizer
from fuyu_utils import PointCloudProcessMixin, VisualInstructionTuningDataset3D
# from models.fuyu_3d import detector
from models.detector_Vote2Cap_DETR.detector import detector
from typing import Tuple, List
from iou3d import box3d_iou, get_minmax_corners, from_minmax_to_xyzhwl, get_3d_box, batch_get_minmax_corners, batch_from_minmax_to_xyzhwl
import multiprocessing as mp

logger = logging.getLogger(__name__)
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class Vote2CapDETRTokenizer(nn.Module):
    def __init__(self, config, dataset_config=DC):
        super().__init__()
        self.config = config
        self.dataset_config = dataset_config
        # dataset_config.num_semcls = 18
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="../SVC/Vote2Cap_DETR_XYZ_COLOR_NORMAL_best_hf.pth",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../SVC/pc_features/scannetv2-vote2cap-feature-new-2.pkl",
        help="Path to save the extracted features",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="box_features",
    )
    args = parser.parse_args() 
    args.output_path = os.path.join(os.path.dirname(__file__), args.output_path).replace(".pkl", f"_{args.feature_type}.pkl")
    print(args)
    return args


class ScanNetDataset(torch.utils.data.Dataset, PointCloudProcessMixin):
    @staticmethod
    def get_scene_path() -> str:
        return "../SVC/scannet_data"

    @staticmethod
    def get_multiview_path() -> str:
        return "data/scannet/scannet_data/enet_feats_maxpool"

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
        scene_file = "data/scannet/meta_data/scannetv2.txt"
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
        instance_bboxes = self.scene_data[scene_id]['instance_bboxes']

        # color -> normal -> (multiview) -> height
        point_cloud = mesh_vertices[:, 0:6]
        point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0

        normals = mesh_vertices[:, 6:9]
        point_cloud = np.concatenate([point_cloud, normals], 1)  # p (50000, 7)

        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        point_cloud, choices = random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        # return {"point_cloud": point_cloud, "scene_id": scene_id}
        return {
            "point_cloud": point_cloud,
            "point_cloud_dims_min": point_cloud_dims_min,
            "point_cloud_dims_max": point_cloud_dims_max,
            "scene_id": scene_id,
            "instance_bboxes": instance_bboxes,
        }


def collate_fn(examples):
    point_cloud = np.stack([e["point_cloud"] for e in examples], axis=0)
    point_cloud_dims_min = np.stack([e["point_cloud_dims_min"] for e in examples], axis=0)
    point_cloud_dims_max = np.stack([e["point_cloud_dims_max"] for e in examples], axis=0)
    return {
        "inputs": [
            torch.tensor(point_cloud, dtype=torch.float32),
            torch.tensor(point_cloud_dims_min, dtype=torch.float32),
            torch.tensor(point_cloud_dims_max, dtype=torch.float32),
        ],
        "scene_id": [e["scene_id"] for e in examples],
        "instance_bboxes": [e["instance_bboxes"] for e in examples]
    }



def worker(prediction, gts):
    iou_matrix = np.zeros(len(gts))
    for j, gt in enumerate(gts):
        iou_matrix[j], _ = box3d_iou(
            get_3d_box(prediction[3:], 0, prediction[:3]), get_3d_box(gt[3:], 0, gt[:3])
        )
    return iou_matrix

def mutual_iou_2(predictions, gts) -> np.ndarray:
    """
    predictions ~ (K1, 8, 3)
    gts ~ (K2, 6)
    returns (K1, K2) matrix
    """
    iou_matrix = np.zeros((len(predictions), len(gts)))
    predictions = batch_get_minmax_corners(predictions)
    predictions = batch_from_minmax_to_xyzhwl(predictions)
    # for i, pred in enumerate(predictions):
    #     for j, gt in enumerate(gts):
    #         # print(i, j, pred.shape, gt.shape)
    #         # pred_minmax = get_minmax_corners(pred)
    #         # pred_xyzhwl = from_minmax_to_xyzhwl(pred_minmax)
    #         iou_matrix[i, j], _ = box3d_iou(
    #             get_3d_box(pred[3:], 0, pred[:3]), get_3d_box(gt[3:], 0, gt[:3])
    #         )
    # use multiprocessing
    with mp.Pool(128) as pool:
        # iou_matrix = pool.map(worker, range(len(predictions)))
        iou_matrix = pool.starmap(worker, [(pred, gts) for pred in predictions])
    iou_matrix = np.stack(iou_matrix, axis=0)

    return iou_matrix
    

def eval_recall(pred_corners, gt_boxes, iou_thresh=0.5):
    pred_corners = pred_corners.cpu().numpy()
    # gt_boxes = gt_boxes.cpu().numpy()
    iou_matrix = mutual_iou_2(pred_corners, gt_boxes)
    recall = (iou_matrix.max(axis=1) > iou_thresh).sum() / iou_matrix.shape[0]
    return recall

def flip_bounding_boxes_to_scene(bbox_corner):
    bbox_corner[..., [1, 2]] = bbox_corner[..., [2, 1]]
    bbox_corner[..., [2]] = -bbox_corner[..., [2]]
    return bbox_corner
            
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = PointNetPPTokenizer(PretrainedConfig(in_channels=132, freeze_pnpp=True))
    model = Vote2CapDETRTokenizer(PretrainedConfig(return_type=args.feature_type, freeze_vote2cap_detr=True, in_channels=7))
    model = model.to(device)
    model.load_pretrained(args.model_path)
    model.eval()

    out_channels = model.out_channels
    logger.info(f"Model loaded from {args.model_path} with output channels: {out_channels}")
    args.output_path = args.output_path.replace(".pkl", f"_{out_channels}d.pkl")

    dataset = ScanNetDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=collate_fn
    )

    features = []
    recalls = defaultdict(list)
    for batch in tqdm(dataloader, total=len(dataloader)):
        with torch.no_grad():
            # batch = batch.to(model.device)
            inputs = [x.to(device) for x in batch["inputs"]]
            # feature, mask = model(inputs)[0:2]
            results = model(inputs)
            feature, mask = results[0:2]
            if len(results) > 2:
                box_corners = results[2]
            else:
                box_corners = None

        for i, scene_id in enumerate(batch["scene_id"]):
            result_dict = {
                "scene_id": scene_id,
                "feature": feature[i].cpu(),
                "mask": mask[i].cpu(),
            }
            if box_corners is not None:
                # result_dict["box_corners"] = box_corners[i].cpu()
                result_dict["box_corners"] = flip_bounding_boxes_to_scene(box_corners[i].cpu())

                # instance_bboxes = batch["instance_bboxes"][i][..., :6]
                # for iou_thresh in [0.25, 0.5]:
                #     recall = eval_recall(box_corners[i], instance_bboxes, iou_thresh=iou_thresh)
                #     recalls[iou_thresh].append(recall)
                    # logger.info(f"Recall@{iou_thresh} for {scene_id}: {recall:.4f}")

            features.append(result_dict)

    for iou_thresh, recall_list in recalls.items():
        logger.info(f"Recall@{iou_thresh}: {np.mean(recall_list):.4f}")

    with open(args.output_path, "wb") as f:
        torch.save(features, f)

if __name__ == "__main__":
    main()