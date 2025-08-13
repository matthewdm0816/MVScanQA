import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch.distributed as dist

import os
import json
import pickle
import numpy as np
import math
from PIL import Image
from typing import List, Tuple, Dict, Union, Optional, Callable
from collections import OrderedDict, defaultdict
import hashlib
try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None
    pass
from utils.pc_utils import random_sampling, rotx, roty, rotz
from iou3d import (
    get_minmax_corners,
    get_3d_box,
    from_minmax_to_corners,
    from_minmax_to_xyzhwl,
)

from parse import parse
import random
from scipy.optimize import linear_sum_assignment
from iou3d import get_3d_box, box3d_iou, get_3d_box_normal, box3d_iou_orthogonal
from copy import deepcopy
import re

import capeval.bleu.bleu as capblue
import capeval.cider.cider as capcider
import capeval.rouge.rouge as caprouge
import capeval.meteor.meteor as capmeteor

from functools import wraps
from dataclasses import dataclass
from utils.random_cuboid import RandomCuboid

import logging
from shapely.geometry import Polygon, MultiPoint
from icecream import ic
import accelerate

logger = logging.getLogger(__name__)

SCALE_DETECTION = 100
INVALID_BBOX = [-100, -100, -100, 1, 1, 1]
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

N_SHOW_CAPTION_SAMPLES = 10

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SVC_PATH = "/root/SVC" # <- change this to your SVC path

def print_once(message):
    if not hasattr(print_once, "printed"):
        print_once.printed = set()
    if message not in print_once.printed:
        print_once.printed.add(message)
        print(message)

def gather_scalar(accelerator, scalar):
    scalar = torch.tensor(scalar).to(accelerator.device)
    return accelerator.gather(scalar).mean().item()


class LogDatasetMixin:
    def __post_init__(self):
        # logger.info(f"Initialized {self.__class__.__name__} dataset <{getattr(self, "name", "")}> with {len(self)} samples")
        logger.info(f"Initialized {self.get_dataset_description()} with {len(self)} samples")

class Singleton(type):
    _instances = {}
    # we are going to redefine (override) what it means to "call" a class
    # as in ....  x = MyClass(1,2,3)
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # we have not every built an instance before.  Build one now.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
            # here we are going to call the __init__ and maybe reinitialize.
            # if hasattr(cls, '__allow_reinitialization') and cls.__allow_reinitialization:
            if getattr(cls, '__allow_reinitialization', False):
                # if the class allows reinitialization, then do it
                instance.__init__(*args, **kwargs)  # call the init again
        return instance

class AverageMeter:
    def __init__(self, 
                 accelerator: Optional[accelerate.Accelerator] = None,
                 report_period: int = 10, 
                 print_fn: Optional[Callable] = print):
        """
        Initialize the AverageMeter with a configurable report period and print function
        
        Args:
            report_period (int): Number of updates before printing average values
            print_fn (Callable, optional): Custom print function, defaults to standard print
        """
        self.values = {}  # Dictionary to store lists of recent values for each name
        self.report_period = report_period
        self.accelerator = accelerator

        self.value_type_cache = {}

        def print_fn_new(*args, **kwargs):
            if self._need_print:
                print_fn(*args, **kwargs)

        self.print_fn = print_fn_new

    def _convert_to_float(self, value: Union[float, int, np.ndarray, torch.Tensor]) -> float:
        """
        Convert input to float, handling various input types
        
        Args:
            value: Input value to convert
        
        Returns:
            float: Converted value
        """
        if isinstance(value, (np.ndarray, torch.Tensor)):
            return float(value.item())
        return float(value)

    def _format_typed_float(self, value: float, type: str) -> str:
        if type == "amount":
            # .2f
            return f"{value:.2f}"

        elif type == "percent":
            return f"{100 * value:.2f}%"
        
        elif type == "integer":
            return f"{int(value)}"


    def _sync_scalar(self, accelerator, scalar):
        if self.accelerator is None:
            logger.warning("No accelerator found, returning scalar as is")
            return scalar
        
        # Convert to tensor and move to device, then gather and return mean
        scalar = torch.tensor(scalar).to(accelerator.device)
        return accelerator.gather(scalar).mean().item()

    def update(self, name: str, value: Union[float, int, np.ndarray, torch.Tensor], type: str="amount"):
        """
        Update a value for a specific name, tracking only recent values
        
        Args:
            name (str): Name of the value to track
            value: Value to add (supports float, int, numpy array, torch tensor)
        """
        # Convert to float
        float_value = self._convert_to_float(value)
        
        # Initialize if name doesn't exist
        if name not in self.values:
            self.values[name] = []
        
        # Add to list of values
        self.values[name].append(float_value)

        self.value_type_cache[name] = type
        
        # Trim to some multiple of the report period
        # self.values[name] = self.values[name][-self.report_period * 10:]
        
        # Check if we have enough values to report
        # if len(self.values[name]) == self.report_period:
        if len(self.values[name]) % self.report_period == 0:
            avg = self.get_avg(name)
            avg = self._sync_scalar(self.accelerator, avg)
            avg = self._format_typed_float(avg, type)
            self.print_fn(f"{name} - Avg: {avg} (last {self.report_period} values)")
    
    def get_avg(self, name: str) -> float:
        """
        Get the current average for a specific name
        
        Args:
            name (str): Name of the value to retrieve
        
        Returns:
            float: Current average, or 0 if no values
        """
        if name not in self.values or not self.values[name]:
            return 0.0
        # return sum(self.values[name]) / len(self.values[name])
        return sum(self.values[name][-self.report_period:]) / self.report_period
    
    def reset(self, name: Optional[str] = None):
        """
        Reset values for a specific name or all names, reporting averages before reset
        
        Args:
            name (str, optional): Name to reset. If None, reset all.
        """
        # skip if no values
        if len(self.values) == 0:
            logger.info("No values to reset, skipping")
            return 
        
        if name is None:
            # print a report begin and end message
            report_begin_string = "-" * 20 + " Report before reset " + "-" * 20
            self.print_fn(report_begin_string)
            # Report averages for all recorded values before clearing
            for key in self.values.keys():
                avg = self.get_avg(key)
                avg = self._sync_scalar(self.accelerator, avg)
                avg = self._format_typed_float(avg, self.value_type_cache[key])
                self.print_fn(f"{key}: {avg}")
            self.values.clear()
            
            self.print_fn("-" * len(report_begin_string))
        elif name in self.values:
            avg = self.get_avg(name)
            avg = self._sync_scalar(self.accelerator, avg)
            avg = self._format_typed_float(avg, self.value_type_cache[name])

            self.print_fn(f"{key}: {avg}")
            self.values[name].clear()

    @property
    def _need_print(self):
        return self.accelerator is None or self.accelerator.is_main_process

    def get_recent_values(self, name: str) -> List[float]:
        """
        Get recent values for a specific name
        
        Args:
            name (str): Name of the values to retrieve
        
        Returns:
            List[float]: Recent values
        """
        return self.values.get(name, [])
    

class FrameCaptionGetter(metaclass=Singleton):
    __allow_reinitialization = False
    # def __init__(self)
    def setup_captions_from_annotation(self, annotations: Union[List[Dict], List[List[Dict]]]):
        if isinstance(annotations[0], list):
            # multiple annotations list in one list
            annotations = sum(annotations, [])

        captions = defaultdict(set)
        for annotation in annotations:
            scene_id = annotation["scene_id"]
            frame_id = annotation["frame_id"]
            captions[f"{scene_id}|{frame_id}"].add(annotation["description"])
        
        captions = {
            k: list(v) for k, v in captions.items()
        }

        logger.info(f"Loaded {sum(len(v) for v in captions.values())} frame captions")
        self.captions = captions

    @staticmethod
    def parse_image_path(image_path: str):
        # image_path: # .../scene0xxx_xx/color/xxxx.jpg
        scene_id, _, frame_id = image_path.split("/")[-3:]
        return scene_id, frame_id
    
    def get_frame_captions(self, scene_id: str, frame_id: str) -> List[str]:
        return self.captions[f"{scene_id}|{frame_id}"]

def resize_image_to_max_weight_height(image, max_weight_height: Tuple[int, int]):
    # keep the aspect ratio
    width, height = image.size
    max_width, max_height = max_weight_height

    # Calculate scaling factor to fit within max dimensions
    width_ratio = max_width / width
    height_ratio = max_height / height
    scale_factor = min(width_ratio, height_ratio)

    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image using LANCZOS resampling for high-quality downscaling
    return image.resize((new_width, new_height), Image.LANCZOS)


def get_intrinsics_and_pose(frame_path, scene_id, frame_id):
    intrinsics_path = os.path.join(frame_path, f"{scene_id}/intrinsic_depth.txt")
    pose_path = os.path.join(frame_path, f"{scene_id}/pose/{frame_id}.txt")
    intrinsics = np.loadtxt(intrinsics_path)
    pose = np.loadtxt(pose_path)
    return intrinsics, pose

def get_scanqa_image_for_question(frame_path, scene_to_image, whole_qid, *args):
    scene_id = whole_qid.split("-")[1]  # train-xxxx-xxxx
    image = scene_to_image[whole_qid][0]  # xxxx.jpg
    image_path = os.path.join(frame_path, f"{scene_id}_00/color/{image}")
    return Image.open(image_path), image_path, get_intrinsics_and_pose(frame_path, f"{scene_id}_00", image.split(".")[0])

def get_scanqa_mv_images_for_question(frame_path, scene_to_image, whole_qid, num_views, *args):
    scene_id = whole_qid.split("_mv")[0]  # scenexxxxxx_mv_xxx
    images = scene_to_image[whole_qid][:num_views]  # xxxx.jpg
    image_paths = [os.path.join(frame_path, f"{scene_id}/color/{image}") for image in images]
    return (
        [Image.open(image_path) for image_path in image_paths], 
        image_paths[0], 
        # [get_intrinsics_and_pose(frame_path, scene_id, image.split(".")[0]) for image in images]
        get_intrinsics_and_pose(frame_path, scene_id, images[0].split(".")[0]) 
        # NOTE: since we only use these camera parameters for framecap dataset to filter objects, any is ok here
        # image_path is used to get frame caption, so we only need the first image_path
    )

def get_sqa3d_image_for_question(frame_path, scene_to_image, whole_qid, scene_id, *args):
    question_id = whole_qid.split("-")[-1]  # sqa3d-xxxxxxx
    image = scene_to_image[question_id][0]  # xxxx.jpg
    # image_path = os.path.join(frame_path, f"{scene_id}/{image}")
    image_path = os.path.join(frame_path, f"{scene_id}/color/{image}")
    return Image.open(image_path).resize((320, 240)), image_path, get_intrinsics_and_pose(frame_path, scene_id, image.split(".")[0])


def get_scanrefer_image_for_instruction(frame_path, scene_to_image, scanrefer_id, *args):
    # scanrefer_id like: 'scanrefer|scene0000_00|15|0|a black tv, in the direction from the entrance and from the outside, will be on the right side of the blue curtain . on the left of the tv is a small bike.'
    dataset_name, scene_id, object_id, ann_id, description = scanrefer_id.split("|")
    # image = scene_to_image[whole_qid][0]  # xxxx.jpg
    image = scene_to_image[scanrefer_id][0]  # xxxx.jpg
    image_path = os.path.join(frame_path, f"{scene_id}/color/{image}")
    return Image.open(image_path), image_path, get_intrinsics_and_pose(frame_path, f"{scene_id}", image.split(".")[0])


def get_scan2cap_image_for_instruction(frame_path, scene_to_image, scene_id, bbox_id, *args):
    # scene_id, _, bbox_id = whole_qid.split("_") # scene0xxx_xx_x
    # scene_id = f"{scene_id}_00" if not scene_id.endswith("_00") else scene_id
    bbox_id = str(bbox_id)
    image = scene_to_image[scene_id][bbox_id]
    if image is None:
        image = "0"  # very rare case, use 0.jpg
    image_path = os.path.join(frame_path, f"{scene_id}/color/{image}.jpg")
    return Image.open(image_path), image_path, get_intrinsics_and_pose(frame_path, scene_id, image)

def get_frame2cap_image_for_instruction(frame_path, scene_id, frame_id, *args):
    image_path = os.path.join(frame_path, f"{scene_id}/color/{frame_id}")
    return Image.open(image_path), image_path, get_intrinsics_and_pose(frame_path, scene_id, frame_id.split(".")[0])

def get_birdview_image(birdview_path, scene_id, *args, **kwargs):
    image_path = os.path.join(birdview_path, f"{scene_id}.png")
    image = Image.open(image_path)
    image = resize_image_to_max_weight_height(image, (512, 512))
    return image, image_path, (np.eye(4), np.eye(4))

def dummy_image_getter(*args, **kwargs):
    return Image.new("RGB", (32, 32), (255, 255, 255)), "_dummy.jpg", (np.eye(4), np.eye(4))


def get_optimizer_param_groups_by_names_dict(
    model: nn.Module,
    names_dict: OrderedDict[str, List[str]],
    lr_dict: Dict[str, float],
    weight_decay_dict: Dict[str, float],
    lr_default: Optional[float] = None,
    weight_decay_default: Optional[float] = None,
) -> Tuple[Dict[str, Dict[str, Union[List[torch.nn.Parameter], float]]], Dict[str, List[str]]]:
    """
    Get optimizer parameter groups by names dict - it shall be OrderedDict to represent the priority.
    unspecifed parameters will be assigned with default values.
    unspecified lr/weight_decay will be assigned with default values or lr_dict["default"]/weight_decay_dict["default"], if the previous is not specified.
    ignores parameters that do not require grad.
    """
    assert (
        lr_default is not None or "default" in lr_dict
    ), "lr_default must be specified or lr_dict must contain a 'default' key"
    assert (
        weight_decay_default is not None or "default" in weight_decay_dict
    ), "weight_decay_default must be specified or weight_decay_dict must contain a 'default' key"
    lr_default = lr_default if lr_default is not None else lr_dict["default"]
    weight_decay_default = (
        weight_decay_default
        if weight_decay_default is not None
        else weight_decay_dict["default"]
    )

    param_groups = {
        param_group_name: {
            "params": [],
            "lr": lr_dict.get(param_group_name, lr_default),
            "weight_decay": weight_decay_dict.get(param_group_name, weight_decay_default),
        }
        for param_group_name in names_dict.keys()
    }
    # add default param group
    param_groups["default"] = {
        "params": [],
        "lr": lr_dict.get("default", lr_default),
        "weight_decay": weight_decay_dict.get("default", weight_decay_default),
    }
    param_names_groups = {param_group_name: [] for param_group_name in param_groups.keys()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        selected = False
        for param_group_name, param_names in names_dict.items():
            if any([param_name in name for param_name in param_names]):
                param_groups[param_group_name]["params"].append(param)
                param_names_groups[param_group_name].append(name)
                selected = True
                break
        if not selected:
            param_groups["default"]["params"].append(param)
            param_names_groups["default"].append(name)

    # remove empty param groups
    param_groups = {
        param_group_name: param_group
        for param_group_name, param_group in param_groups.items()
        if len(param_group["params"]) > 0
    }

    return param_groups, param_names_groups


def mutual_iou(predictions, gts) -> np.ndarray:
    """
    predictions ~ (K1, 6), xyzhwl
    gts ~ (K2, 6)
    """
    iou_matrix = np.zeros((len(predictions), len(gts)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(gts):
            # iou_matrix[i, j], _ = box3d_iou(
            #     get_3d_box(pred[3:], 0, pred[:3]), get_3d_box(gt[3:], 0, gt[:3])
            # )
            iou_matrix[i, j] = box3d_iou_orthogonal(
                pred[:6], gt[:6]
            )

    return iou_matrix
    

def assign_preds_to_gts(predictions, gts) -> Tuple[np.ndarray, np.ndarray]:
    """
    predictions ~ (K1, 6), (x,y,z, size_x, size_y, size_z)
    gts ~ (K2, 6)
    """
    iou_matrix = mutual_iou(predictions, gts)

    # no need of hungarian algorithm, since we can have duplicate assignments
    pred_idx_assigned, pred_iou = np.argmax(iou_matrix, axis=0), np.max(
        iou_matrix, axis=0
    )  # (K2, ), (K2, )
    return pred_idx_assigned, pred_iou

def assign_gts_to_preds(predictions, gts) -> Tuple[np.ndarray, np.ndarray]:
    """
    predictions ~ (K1, 6), xyzhwl
    gts ~ (K2, 6)
    """
    iou_matrix = mutual_iou(predictions, gts)

    # no need of hungarian algorithm, since we can have duplicate assignments
    gt_idx_assigned, gt_iou = np.argmax(iou_matrix, axis=1), np.max(
        iou_matrix, axis=1
    )  # (K1, ), (K1, )
    return gt_idx_assigned, gt_iou


class PointCloudProcessMixin:
    """
    Point cloud augmentation helper functions
    """

    def _augment_pc(
        self, point_cloud, instance_bboxes, do_rotx=True, do_roty=True, do_rotz=True, do_translate=True
    ):
        """
        Augment partial point cloud and bounding boxes
        instance_bboxes ~ (N, 6), xyzhwl
        """
        # ------------------------------- DATA AUGMENTATION ------------------------------
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            instance_bboxes[:, 0] = -1 * instance_bboxes[:, 0]
            flip_x = -1
        else:
            flip_x = 1

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            instance_bboxes[:, 1] = -1 * instance_bboxes[:, 1]
            flip_y = -1
        else:
            flip_y = 1

        # Rotation along X-axis
        if do_rotx:
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        else:
            rot_angle = 0

        rot_mat = rotx(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

        rot_mat_total = rot_mat[:]  # Rx

        # Rotation along Y-axis
        if do_roty:
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        else:
            rot_angle = 0
        rot_mat = roty(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

        rot_mat_total = np.dot(rot_mat, rot_mat_total)  # RyRx

        # Rotation along up-axis/Z-axis
        if do_rotz:
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
        else:
            rot_angle = 0
        rot_mat = rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))

        # FIXME: this only process the z-axis rotation!!! (as we only rotate z-axis)
        instance_bboxes = self.rotate_aligned_boxes(instance_bboxes, rot_mat)

        rot_mat_total = np.dot(rot_mat, rot_mat_total)  # RzRyRx


        # Translation
        if do_translate:
            point_cloud, factor = self._translate(point_cloud)
            instance_bboxes[:, :3] += factor
        else:
            factor = [0.0, 0.0, 0.0]

        return point_cloud, instance_bboxes, factor, rot_mat_total, flip_x, flip_y

    def _translate(self, point_set):
        # unpack
        coords = point_set[:, :3]
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        # dump
        coords += factor
        point_set[:, :3] = coords

        return point_set, factor

    @staticmethod
    def normalize_color(color: np.ndarray, is_color_in_range_0_255: bool = False) -> np.ndarray:
        r"""
        Convert color in range [0, 1] to [-0.5, 0.5]. If the color is in range [0,
        255], use the argument `is_color_in_range_0_255=True`.

        `color` (torch.Tensor): Nx3 color feature matrix
        `is_color_in_range_0_255` (bool): If the color is in range [0, 255] not [0, 1], normalize the color to [0, 1].
        """
        if is_color_in_range_0_255:
            color /= 255
        color -= 0.5
        # return color.float()
        return color.astype(np.float32)
    
    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class VisualInstructionTuningDataset3D(Dataset, PointCloudProcessMixin, LogDatasetMixin):
    """
    Base class for visual instruction tuning dataset, equipped with 3D point cloud
    sample keys:
        [a, b, c...] (text instructions)
        "prompt" (combines all instructions)
        "image" (paired image)
        "coords" (3D point cloud coords, MinkowskiEngine format)
        "feats" (3D point cloud feats, MinkowskiEngine format)
        "labels" (3D point cloud labels, MinkowskiEngine format)
        [other informative keys: "question_id", "scene_id"] question_id is a generic id (not only for ScanQA)
    """

    def get_annotation_file(self, split="train") -> str:
        raise NotImplementedError

    @staticmethod
    def get_scene_path() -> str:
        return f"{SVC_PATH}/scannet_data"

    @staticmethod
    def get_multiview_path() -> str:
        return f"{DATA_PATH}/scannet/scannet_data/enet_feats_maxpool"

    @staticmethod
    def get_frozen_object_feature_path(type: str="pnpp") -> str:
        if type == "pnpp":
            return f"{DATA_PATH}/scannetv2-pnpp-feature.pkl"
        elif type == "pnpp-vote2cap-box":
            # return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature_box_features_281d.pkl" # its box need flip!
            return f"{SVC_PATH}/pc_features/scannetv2-vote2cap-feature-new-2_box_features_281d.pkl" # this don't
        elif type == "uni3d-mask3d-box":
            return f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt" # 1030d
        elif type == "pnpp-vote2cap-enc":
            return f"{DATA_PATH}/scannetv2-vote2cap-feature_enc_features_259d.pkl"
        else:
            raise ValueError(f"Unknown frozen object feature type: {type}")

    def __len__(self):
        return len(self.annotation)

    def get_all_scene_ids(self):
        if not hasattr(self, "scene_ids"):
            import glob

            # load all scene_id in scene_path
            scene_path = self.get_scene_path()
            filenames = glob.glob(
                os.path.join(scene_path, "*_aligned_vert.npy")
            )  # .../scene0804_00_aligned_vert.npy
            self.scene_ids = [
                os.path.basename(f)[: -len("_aligned_vert.npy")] for f in filenames
            ]
            logger.info(f"Loaded {len(self.scene_ids)} scene ids")
            return self.scene_ids
        else:
            return self.scene_ids
        
    def get_dataset_description(self):
        return f"{self.__class__.__name__}-{self.name}-{self.split}"

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        use_color: bool,
        use_height: bool,
        use_normal: bool,
        use_multiview: bool,
        use_augment: bool,
        i2t: str = None,
        views_path: str = None,
        prompt: str = None,
        instruction_keys: List[str] = None,
        image_getter: callable = None,
        quantization_size: float = 0.02,
        num_points: int = 50_000,
        reinforce_bbox: bool = False,
        shift_bbox_to_positive: bool = False,
        prompt_end_token: str = None,
        use_random_cuboid: bool = False,
        random_cuboid_min_points: int = 30_000,
        framecap_as_input: bool = False,
        scale_bbox: int = 100, # -1 for no scale and use float bbox
        use_llm_style_prompt: bool = False,
        frozen_object_type: Optional[str] = None,
        shuffle_objects: bool = False,
        start_from_last: bool = False,
        remove_bad_start_words: bool = True,
        enforce_validation: bool = False,
        multiple_input_images: str = "1x1",
        use_birdview: bool = False,
        birdview_path: Optional[str] = None,
        use_object_index: bool = False,
        **kwargs,
    ):
        self.name: str = name
        self.split: str = split
        self.annotation = self.get_annotation_file(split)
        self.start_from_last = start_from_last
        self.accessed_times = defaultdict(int)
        if isinstance(self.annotation, str):
            self.annotation = json.load(open(self.get_annotation_file(split)))
        else:
            assert isinstance(self.annotation, list) # already a (loaded from JSON) list
        self.prompt_end_token = (
            prompt_end_token if prompt_end_token is not None else "|ENDOFTEXT|"
        )
        if ratio < 1.0:
            # self.annotation = self.annotation[: int(len(self.annotation) * ratio)]
            self._take_partial_data(ratio)


        # self.i2t = json.load(open(i2t))["view"]
        # check .pkl or .json
        if isinstance(i2t, str) and (i2t.endswith(".pkl") or i2t.endswith(".pth")):
            self.i2t = torch.load(i2t, map_location="cpu")["view"]
        else:
            self.i2t = json.load(open(i2t))
            if self.name in ["scanrefer", "scanqa", "sqa3d"] and "view" in self.i2t:
                self.i2t = self.i2t["view"]

        self.views_path = views_path
        self.prompt = prompt

        self.frame_caption_prompt = [
            "The 2D view of the room shows: {frame_caption}\n",
            "The 2D view of the room displays: {frame_caption}\n",
            "Here is the simple description of the 2D view: {frame_caption}\n",
            "Context in 2D view: {frame_caption}\n",
            "A brief description of the 2D view: {frame_caption}\n",
        ]

        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_augment = use_augment

        self.image_getter = (
            image_getter  # receives (views_path, i2t, whole_qid) and returns PIL.Image
        )
        self.instruction_keys = instruction_keys

        self.quantization_size = quantization_size
        self.num_points = num_points
        self.reinforce_bbox = reinforce_bbox
        self.shift_bbox_to_positive = shift_bbox_to_positive

        self.pc_tokenizer_type = "minkowski"
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.use_random_cuboid = use_random_cuboid
        
        self.framecap_as_input = framecap_as_input
        self.scale_bbox = scale_bbox
        self.use_llm_style_prompt = use_llm_style_prompt

        self.frozen_object_type = frozen_object_type
        self.shuffle_objects = shuffle_objects
        self.remove_bad_start_words = remove_bad_start_words
        self.enforce_validation = enforce_validation
        self.multiple_input_images = multiple_input_images
        self.use_birdview = use_birdview
        self.birdview_path = birdview_path
        self.use_object_index = use_object_index

        self._preprocess_annotation()  # preprocess annotation
        self._load()  # load scene data
        self.__post_init__()

    def set_pc_tokenizer_type(self, pc_tokenizer_type: str):
        self.pc_tokenizer_type = pc_tokenizer_type
        if pc_tokenizer_type == "frozen":
            self._load_frozen_features(self.frozen_object_type)
        elif pc_tokenizer_type == "merged-frozen":
            self._load_merged_frozen_features(self.frozen_object_type.split("+"))

    def _take_partial_data(self, ratio: float):
        """
        Take partial data from the annotation
        """
        # self.annotation = self.annotation[: int(len(self.annotation) * ratio)]
        if self.start_from_last:
            # take from last X percent
            self.annotation = self.annotation[-int(len(self.annotation) * ratio) :]
        else:
            self.annotation = self.annotation[: int(len(self.annotation) * ratio)]

    def _preprocess_annotation(self):
        pass

    def _splice_multiview_image(self, images, multiview_shape: Optional[str]=None):
        if multiview_shape is None:
            multiview_shape = self.multiple_input_images

        h, w = multiview_shape.split("x") # 1x1, 2x2, 1x4, etc.
        h, w = int(h), int(w)

        # Validate input
        if len(images) != h * w:
            raise ValueError(f"Number of images ({len(images)}) does not match grid shape {h}x{w}")

        if len(images) == 1:
            # print_once("Only one image, returning as is. Is this intended?")
            return images[0]

        # Get dimensions of individual images
        img_width, img_height = images[0].size

        # Create a blank canvas for the spliced image
        spliced_image = Image.new(images[0].mode, (img_width * w, img_height * h))

        # Populate the spliced image
        for idx, img in enumerate(images):
            row = idx // w
            col = idx % w
            
            position = (col * img_width, row * img_height)
            spliced_image.paste(img, position)

        return spliced_image

    def _load(self):
        """
        Load 3D scene, instance and object information
        """
        logger.info("Loading scene (ScanNet) data...")
        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.annotation])))
        logger.info(f"Loaded {len(self.scene_list)} scenes")
        logger.info(self.scene_list)
        # self.scene_list = self.get_all_scene_ids()

        # load scene data
        self.scene_data = {}
        scene_path = self.get_scene_path()
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_vert.npy"
            )  # axis-aligned
            self.scene_data[scene_id]["instance_labels"] = np.load(
                os.path.join(scene_path, scene_id) + "_ins_label.npy"
            )
            self.scene_data[scene_id]["semantic_labels"] = np.load(
                os.path.join(scene_path, scene_id) + "_sem_label.npy"
            )
            self.scene_data[scene_id]["instance_bboxes"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_bbox.npy"
            )
            try:
                axis_align_matrix = json.load(open("data/alignments.json", "r"))[scene_id]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            except KeyError:
                axis_align_matrix = np.eye(4) # for test scenes
            self.scene_data[scene_id]["axis_align_matrix"] = axis_align_matrix

        self.input_predicted_bboxes = self.get_input_predicted_bbox()
        self._compute_closest_predicted_bbox()


    def get_input_predicted_bbox(self) -> Dict[str, np.ndarray]:
        """
        assumed to return a scene_id -> list of predicted bboxes mapping
        no masked predicted bboxes!
        """
        if self.frozen_object_type == "pnpp-vote2cap-box":
            predicted_bbox_file = f"{SVC_PATH}/pc_features/scene_bbox_info_for_valtest_vote2cap_detr.pkl"
            logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")

            predicted_bbox = pickle.load(open(predicted_bbox_file, "rb"))
            # scene_id -> {bbox_id: {"bbox": ..., "is_valid": True/False}, ...}
            # NOTE: the predicted bboxes are already trimmed, so the index is the index among valid predicted bboxes
            for scene_id in predicted_bbox:
                # predicted_bbox[scene_id] = np.stack(
                #     [np.array(item["bbox"]) for item in predicted_bbox[scene_id] if item["is_valid"]]
                # )
                num_bboxes = len(predicted_bbox[scene_id])
                bboxes = np.zeros((num_bboxes, 6))
                for bbox_id, item in predicted_bbox[scene_id].items():
                    # print(item)
                    assert item["is_valid"]
                    if item["is_valid"]:
                        bboxes[bbox_id] = np.array(item["bbox"])

                predicted_bbox[scene_id] = bboxes

        elif self.frozen_object_type == "uni3d-mask3d-box":
            predicted_bbox_file = f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt" # 1030d
            logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")
            predicted_bbox_anno = torch.load(predicted_bbox_file, map_location="cpu") # a list of dict

            # note that all of them are considered valid
            predicted_bbox = {}
            for item in predicted_bbox_anno:
                scene_id = item["scene_id"]
                bboxes = item["bbox"]
                predicted_bbox[scene_id] = bboxes.numpy() # [N_bboxes, 6]

            for scene_id in self.scene_list:
                if scene_id not in predicted_bbox:
                    logger.warning(f"Scene {scene_id} has no predicted bboxes!")
                    # predicted_bbox[scene_id] = np.zeros((100, 6))
                    predicted_bbox[scene_id] = np.random.rand(100, 6)

        else:
            raise NotImplementedError(f"Invalid frozen object type: {self.frozen_object_type} that have no predicted bboxes")
        
        return predicted_bbox

    def _compute_closest_predicted_bbox(self):
        """
        calculate all IoU3D between GT bbox and predicted bboxes, find the closest one for each GT bbox
        """
        for scene_id in self.scene_list:
            pred_bboxes = self.input_predicted_bboxes[scene_id][:, :6]  # (K1, 6)
            # pred_bboxes_object_ids = self.input_predicted_bboxes[scene_id][:, -1]  # (K1,)
            pred_bboxes_object_ids = np.arange(len(pred_bboxes))  # (K1,)
            gt_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]  # (K2, 6)
            gt_object_ids = self.scene_data[scene_id]["instance_bboxes"][:, -1]  # (K2,)
            
            
            pred_idx_assigned, pred_iou = assign_preds_to_gts(pred_bboxes, gt_bboxes)  # (K2,), (K2,)
            for i, gt_object_id in enumerate(gt_object_ids):
                gt_object_id = int(gt_object_id)
                pred_id = int(pred_bboxes_object_ids[pred_idx_assigned[i]]) # assumably pred_idx_assigned[i], since pred_bboxes_object_ids is a range
                iou = pred_iou[i]
                
                # Store the closest predicted bbox and its IoU for each GT bbox
                if "closest_pred_bbox" not in self.scene_data[scene_id]:
                    self.scene_data[scene_id]["closest_pred_bbox"] = {}
                self.scene_data[scene_id]["closest_pred_bbox"][gt_object_id] = {
                    "pred_id": pred_id,
                    "iou": iou
                }

    def _load_frozen_features(self, frozen_object_type: str):
        if not hasattr(self, "frozen_features"):
            frozen_features_path = self.get_frozen_object_feature_path(frozen_object_type)
            logger.info(f"Loading frozen features from {frozen_features_path}...")
            self.frozen_features = torch.load(frozen_features_path, map_location="cpu")
            self.frozen_features = {
                item["scene_id"]: (item["feature"], item["mask"], item["box_corners"]) if "box_corners" in item else (item["feature"], item["mask"])
                for item in self.frozen_features
            }
            self.frozen_in_channels = next(iter(self.frozen_features.values()))[0].shape[1]

            logger.info(f"loaded {frozen_object_type} object features")

    def _load_merged_frozen_features(self, used_frozen_object_types: List[str]):
        if not hasattr(self, "merged_frozen_features"):
            self.merged_frozen_features = {}
            # self.merged_frozen_in_channels = {}
            self.merged_frozen_in_channels = []
            for frozen_object_type in used_frozen_object_types:
                frozen_features_path = self.get_frozen_object_feature_path(frozen_object_type)
                logger.info(f"Loading frozen features from {frozen_features_path}...")
                frozen_features = torch.load(frozen_features_path, map_location="cpu")
                frozen_features = {
                    item["scene_id"]: (item["feature"], item["mask"], item["box_corners"]) if "box_corners" in item else (item["feature"], item["mask"])
                    for item in frozen_features
                }
                self.merged_frozen_features[frozen_object_type] = frozen_features
                self.merged_frozen_in_channels.append(next(
                    iter(frozen_features.values())
                )[0].shape[1])

                logger.info(f"loaded {frozen_object_type} object features")
            logger.info(f"merged frozen in channels: {self.merged_frozen_in_channels}")

    def __getitem__(self, idx):
        self.accessed_times[idx] += 1
        # --- get textual data and ids ---
        data_type = self.name
        split = self.split
        scene_id = self.annotation[idx]["scene_id"]
        ann_id = self.annotation[idx].get("ann_id", "-1")
        question_id = f"{scene_id}_{ann_id}"  # for scanrefer/scan2cap
        if "sr3d" in self.name.lower():
            question_id = f"{scene_id}_{ann_id}"
            _description = self.annotation[idx]["description"]
            # hash description, take first 6 characters, since there are multiple descriptions for single ann_id
            question_id += hashlib.md5(_description.encode()).hexdigest()[:6]

        raw_question_id = self.annotation[idx].get("question_id", question_id)
        raw_question_id = str(raw_question_id)

        

        # get scan2cap corpus id
        try:
            object_name = self.annotation[idx]["object_name"]
            object_id = int(self.annotation[idx]["object_id"])
        except KeyError as e:
            try:
                object_name = self.annotation[idx]["object_names"][0]
                object_id = int(self.annotation[idx]["object_ids"][0])
            except (KeyError, IndexError) as e:
                object_name = "unknown"  # sqa3d has no object name
                object_id = 0

        object_id = int(object_id) if object_id is not None else 0
        scan2cap_id = f"{scene_id}|{object_id}|{object_name}"

        object_name = object_name.replace("_", " ")  # for instruction, replace _ with space

        if "scanrefer" in self.name.lower():
            description = self.annotation[idx]["description"]
            if "nr3d" in self.name.lower():
                dataset_name = "nr3d"
            elif "sr3d" in self.name.lower():
                dataset_name = "sr3d"
            else:
                dataset_name = "scanrefer"

            scanrefer_id = f"{dataset_name}|{scene_id}|{object_id}|{ann_id}|{description}"
        else:
            scanrefer_id = None


        # --- get scene data ---
        mesh_vertices = self.scene_data[scene_id][
            "mesh_vertices"
        ].copy()  # supposedly (50000, 9), xyz, rgb, normal
        instance_labels = self.scene_data[scene_id]["instance_labels"].copy()
        semantic_labels = self.scene_data[scene_id]["semantic_labels"].copy()
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"].copy()  # xyzhwl

        if self.pc_tokenizer_type == "frozen":
            self._load_frozen_features(self.frozen_object_type)
            # object_feature, object_mask = self.frozen_features[scene_id]
            object_feature = self.frozen_features[scene_id][0]
            object_mask = self.frozen_features[scene_id][1]
            if object_mask.sum() == 0:
                logger.warning(f"Empty mask for {scene_id}")
                object_mask = torch.ones_like(object_mask)

            predicted_bbox_corners = self.frozen_features[scene_id][2] if len(self.frozen_features[scene_id]) == 3 else None

        if self.pc_tokenizer_type == "merged-frozen":
            self._load_merged_frozen_features(self.frozen_object_type.split("+"))
            object_feature = []
            object_mask = []
            predicted_bbox_corners = []
            for frozen_object_type in self.frozen_object_type.split("+"):
                # feat, mask = self.merged_frozen_features[frozen_object_type][scene_id]
                feat = self.merged_frozen_features[frozen_object_type][scene_id][0]
                mask = self.merged_frozen_features[frozen_object_type][scene_id][1]
                box = self.merged_frozen_features[frozen_object_type][scene_id][2] if len(self.merged_frozen_features[frozen_object_type][scene_id]) == 3 else None
                object_feature.append(feat)
                object_mask.append(mask)
                predicted_bbox_corners.append(box)

        mesh_vertice_min = mesh_vertices[:, :3].min(axis=0)  # xyz

        # color -> normal -> (multiview) -> height
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            if self.pc_tokenizer_type == "minkowski":
                point_cloud[:, 3:6] = point_cloud[:, 3:6] / 256.0
            elif self.pc_tokenizer_type in ["pointnet++", "vote2cap-detr"]:
                point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            elif self.pc_tokenizer_type == "frozen":
                ... # DO nothing
            pcl_color = point_cloud[:, 3:6]

        if self.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)  # p (50000, 7)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # load multiview database
        if self.use_multiview:
            enet_feats_file = os.path.join(self.get_multiview_path(), scene_id) + ".pkl"
            multiview = pickle.load(open(enet_feats_file, "rb"))
            point_cloud = np.concatenate([point_cloud, multiview], 1)  # p (50000, 135)
            # print(point_cloud.shape)

        point_cloud, choices = random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        instance_labels = instance_labels[choices]  # not used
        semantic_labels = semantic_labels[choices]  # not used
        pcl_color = pcl_color[choices]

        shuffle_indices = None
        shuffled_indices_trimmed = None

        if self.use_augment and self.split == "train":
            rotx = (
                self.pc_tokenizer_type != "vote2cap-detr"
            )  # vote2cap-detr only rotate on z-axis, and no translation
            point_cloud, instance_bboxes, factor, rot_mat, flip_x, flip_y = self._augment_pc(
                point_cloud, instance_bboxes, do_rotx=rotx, do_roty=rotx, do_translate=rotx
            )
        else:
            factor = [0, 0, 0]
            rot_mat = np.eye(3)
            flip_x = 1
            flip_y = 1

        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        #   |-- give 3D data to specific tokenizer
        #   |-- Minowski Engine
        if self.pc_tokenizer_type == "minkowski":
            coords, feats, labels = ME.utils.sparse_quantize(
                coordinates=point_cloud[:, :3],
                features=self.normalize_color(point_cloud[:, 3:], is_color_in_range_0_255=True),
                labels=semantic_labels,
                quantization_size=self.quantization_size,
                ignore_label=-100,
            )
            pc_dict = {
                "coords": coords,
                "feats": feats,
                "labels": labels,
            }
        #  |-- PointNet++
        elif self.pc_tokenizer_type == "pointnet++":
            pc_dict = {
                "point_cloud": point_cloud,
            }
        #  |-- Frozen features
        elif self.pc_tokenizer_type == "frozen":
            object_mask_np = object_mask.bool().numpy() if isinstance(object_mask, torch.Tensor) else object_mask
            if object_mask_np.sum() == 0:
                logger.warning(f"Object mask is all-false in scene {scene_id}!")
                object_mask_np = np.ones(len(object_feature), dtype=bool)

            if self.shuffle_objects and self.split == "train" and not self.enforce_validation:
                # shuffle object features
                # FIXME: how to let target label also shuffle?
                # 1, 2, 3, .., => 3, 1, 2, ... (shuffle) => 3, ..., 17, ..., 8 (shuffle, valid) => 1, .., 3, ... 2 (shuffle, valid, in trimmed objects)
                # np.random.seed(idx + self.accessed_times[idx])
                generator = np.random.default_rng(seed=idx + self.accessed_times[idx])
                    # ensure different seed for same index at different time
                # shuffle_indices = np.random.permutation(len(object_feature))
                shuffle_indices = generator.permutation(len(object_feature))

                shuffled_indices_trimmed = np.argsort(shuffle_indices[object_mask_np]) #　0-th goes to shuffled_indices_trimmed[0]-th, 1-th goes to shuffled_indices_trimmed[1]-th, ...

                object_feature = object_feature[shuffle_indices] 
                    # shuffle_indices[0]-th -> 0-th, shuffle_indices[1]-th -> 1-th, ...
                    # so 0-th -> argsort(shuffle_indices)[0]-th, 1-th -> argsort(shuffle_indices)[1]-th, ...
                object_mask = object_mask[shuffle_indices]
                predicted_bbox_corners = predicted_bbox_corners[shuffle_indices]

            else:
                shuffled_indices = np.arange(len(object_feature)) # no shuffle, 0, 1, ..., N_obj-1
                shuffled_indices_valid = shuffled_indices[object_mask_np]
                # shuffled_indices_trimmed = np.argsort(np.argsort(shuffled_indices_valid)) # always ascending, 0, 1, ..., N_valid-1
                shuffled_indices_trimmed = np.argsort(shuffled_indices_valid)
                # if np.all(object_mask_np):
                assert np.allclose(shuffled_indices_trimmed, np.arange(len(shuffled_indices_trimmed))), f"Something wrong with shuffled_indices_trimmed: {shuffled_indices_trimmed}"

            pc_dict = {
                "object_feature": object_feature,
                "object_mask": object_mask,
                "predicted_bbox_corners": predicted_bbox_corners,
            }
        # |-- Vote2Cap-DETR
        elif self.pc_tokenizer_type == "vote2cap-detr":
            pc_dict = {
                "point_cloud": point_cloud,
                "point_cloud_dims_min": point_cloud_dims_min,
                "point_cloud_dims_max": point_cloud_dims_max,
            }
        elif self.pc_tokenizer_type == "merged-frozen":
            pc_dict = {
                "object_feature": object_feature,
                "object_mask": object_mask,
                "merged-frozen": True,
                # "predicted_bbox_corners": predicted_bbox_corners,
                # since some are box, some are points, and some are voxels.
            }

        # --- get target bbox ---
        unique_instance_ids = instance_bboxes[:, -1]
        target_id = 0
        for i, unique_instance_id in enumerate(unique_instance_ids):
            if int(unique_instance_id) == int(object_id):
                target_id = i
                break

        # --- get related bbox ---
        related_object_bboxes = []
        gt_instance_bboxes = instance_bboxes.copy()
        # if instance_bboxes_gt exists, use it
        if "instance_bboxes_gt" in self.annotation[idx]:
            gt_instance_bboxes = np.array(self.annotation[idx]["instance_bboxes_gt"])

        if self.name == "scanqa" and "test" not in self.split:
            # use related object bboxes for ScanQA, only for train and val
            related_object_ids = deepcopy(self.annotation[idx]["object_ids"])
            for i, unique_instance_id in enumerate(unique_instance_ids):
                if int(unique_instance_id) in related_object_ids:
                    related_object_bboxes.append(gt_instance_bboxes[i, 0:6].copy())
            # shuffle
            # random.shuffle(related_object_bboxes)
        
        if "scan2cap" in self.name: # for all scan2cap datasets
            # use ALL gt bboxes as related bboxes
            related_object_bboxes = gt_instance_bboxes[...,0:6].copy()
            # random.shuffle(related_object_bboxes)

        related_object_bboxes = np.array(related_object_bboxes)
        

        # --- get image ---
        if self.use_birdview:
            image, image_path, (intrinsics, poses) = get_birdview_image(self.birdview_path, scene_id)

        elif "scan2cap" in self.name or "scan2obj" in self.name:
            image, image_path, (intrinsics, poses) = self.image_getter(
                self.views_path, self.i2t, scene_id, target_id
            )
        elif "scanrefer" in self.name:
            image, image_path, (intrinsics, poses) = self.image_getter(
                self.views_path, self.i2t, scanrefer_id
            )
        elif self.name == "scanqa":
            image, image_path, (intrinsics, poses) = self.image_getter(self.views_path, self.i2t, raw_question_id)
        elif self.name == "sqa3d":
            image, image_path, (intrinsics, poses) = self.image_getter(
                self.views_path, self.i2t, "sqa3d-" + raw_question_id, scene_id
            )
        elif self.name in ["lamm3d", "scenecap"]:
            image, image_path, (intrinsics, poses) = self.image_getter()  # for dummy
        elif self.name in ["framecap", "frameqa"]:
            frame_id = self.annotation[idx]["frame_id"]
            image, image_path, (intrinsics, poses) = self.image_getter(self.views_path, scene_id, frame_id)
        elif self.name in ["scanqa-mv"]:
            h, w = self.multiple_input_images.split("x")
            num_views = int(h) * int(w)
            images, image_path, (intrinsics, poses) = self.image_getter(self.views_path, self.i2t, raw_question_id, num_views)
            image = self._splice_multiview_image(images, multiview_shape=self.multiple_input_images)
        else:
            raise NotImplementedError
        
        axis_align_matrix = self.scene_data[scene_id]["axis_align_matrix"]
        frame_caption_mask = int(data_type in ["framecap", "frameqa"])

        #   |-- get frame caption
        try:
            scene_id_maybe, frame_id_maybe = FrameCaptionGetter.parse_image_path(image_path)
            frame_caption = FrameCaptionGetter().get_frame_captions(scene_id_maybe, frame_id_maybe)
            if self.split != "train":
                frame_caption = frame_caption[0]
            else:
                frame_caption = random.choice(frame_caption)
        except Exception as e:
            frame_caption = ""

        # --- get instruction ---
        #   |-- get target bbox text ---
        target_bbox = instance_bboxes[target_id, 0:6].copy()  # xyzhwl

        if self.shift_bbox_to_positive:
            # shift bbox to positive
            target_bbox[:3] -= mesh_vertice_min  #

        target_corners = get_3d_box(target_bbox[3:6], 0, target_bbox[0:3])

        if self.scale_bbox > 0:
            # print(self.scale_bbox)
            target_bbox_int = np.round(target_bbox * self.scale_bbox).astype(np.int32)
            target_bbox_text = ",".join([f"{x}" for x in target_bbox_int])
            target_bbox_text = f"[{target_bbox_text}]"

        else:
            target_bbox_text = ",".join([f"{x:.2f}" for x in target_bbox])
            target_bbox_text = f"[{target_bbox_text}]"


        #   |-- get textual instructions ---
        #       NOTE: missing keys are filled with empty string
        instructions = {
            key: self.annotation[idx][key] if key in self.annotation[idx] else ""
            for key in self.instruction_keys
        }
        instructions["location"] = target_bbox_text
        instructions["object_index"] = object_id
        instructions["scene_id"] = scene_id

        #  --- change object_index according to shuffled object features ---
        instructions = self.process_instructions(instructions, shuffled_indices_trimmed=shuffled_indices_trimmed if self.shuffle_objects and self.split == "train" and not self.enforce_validation else None)

        prompt = self.prompt if isinstance(self.prompt, str) else np.random.choice(self.prompt)
        target = prompt.format(**instructions)
        
        # remove duplicate ".", "," or continuous spaces
        target = re.sub(r'([., ])\1+', r'\1', target)
        

        if self.framecap_as_input and frame_caption != "":
            # prepend frame caption
            used_framecap_prompt = np.random.choice(self.frame_caption_prompt) if self.split == "train" else self.frame_caption_prompt[0]
            target = f"{used_framecap_prompt.format(frame_caption=frame_caption)}{target}"

        target_instruction = target.split("\x04")[0] # + '\x04' # remove gt outputs for predictions
        
        if self.use_llm_style_prompt:
            target = target.replace('\x04', '[/INST]')
            target = '[INST] ' + target
            target_instruction = target.split('[/INST]')[0] + '[/INST]'

        if self.split != "train" or self.enforce_validation:
            target = target_instruction

        
        return {
            # 2D instruction, image, target
            "question_id": question_id,
            "raw_question_id": raw_question_id,
            "scan2cap_id": scan2cap_id,
            "scene_id": scene_id,
            "scanrefer_id": scanrefer_id,
            "image": image,
            "image_path": image_path,
            "target": target,
            "target_instruction": target_instruction,
            "target_id": target_id, # index in GT bboxes
            "data_type": data_type,
            **instructions,
            "split": split,
            "target_bbox": target_bbox,
            "related_object_bboxes": related_object_bboxes,
            # frame parameters
            "frame_intrinsics": intrinsics,
            "frame_poses": poses,
            "axis_alignments": axis_align_matrix,
            "frame_caption_mask": frame_caption_mask,
            "frame_caption": frame_caption,
            # 3D
            **pc_dict,
        }

    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        """
        Process instructions, e.g. remove template, add prompt, etc.
        """
        return instructions


class ScanReferDataset(VisualInstructionTuningDataset3D):
    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )
        """
        Current design: LVLM predict the index of the object (input predicted bbox & feature)
        At training time, the label is the closest predicted bbox to the GT bbox
        At eval time, get that index and get the corresponding bbox from prediction, calc IoU3D with GT bbox
        Note: we trim objects in forward pass, so index (predicted bbox) is the index among valid predicted bboxes
        """

        # set ScanRefer-specific parameters
        if kwargs.get("prompt", None) is None:
            # postfix = "\x04 {object_index} ." + f"{self.prompt_end_token}"
            postfix = "\x04 <OBJ{object_index}>." + f"{self.prompt_end_token}"
            self.prompt = [
                # "Find in the 3D scene: {description}. X-Y-Z-H-W-L location:",
                # "Find the object in the 3D scene: {description}. X-Y-Z-H-W-L location:",
                "Find the object in the room: \"{description}\". Object index: ",
                "In the 3D scene, find: \"{description}\". Object index: ",
                "The object index of \"{description}\" in the 3D scene is: "
                # "In the 3D scene, find: {description}. X-Y-Z-H-W-L location:",
                # "Description: {description}. The X-Y-Z-H-W-L location in the 3D scene is:",
                # "The xyzhwl place of {description} in the 3D scene is: ",
                # "The xyzhwl location of {description} in the 3D scene is: ",
            ]

            self.prompt = [prompt + postfix for prompt in self.prompt]

        self.instruction_keys = ["description", "object_index"]
        self.instruction_keys_for_prompt = ["description"]
        self.image_getter = get_scanrefer_image_for_instruction

        if self.use_object_index: # cheat to debug object index
            logger.info(f"Using object index in prompt for {self.get_dataset_description()}...")
            self.prompt = [prompt.replace("{description}", "<OBJ{object_index}> {description}") for prompt in self.prompt]
            self.instruction_keys_for_prompt.append("object_index")

        self.start_from_last = kwargs.get("start_from_last", False)

    def _format_object_index(self, object_index: Union[int, str]) -> str:
        if isinstance(object_index, int):
            object_index = str(object_index)

        return f"<OBJ{object_index}>"

    def _parse_object_index(self, caption: str) -> int:
        try:
            object_index = caption.replace(".", "").strip()
            return int(object_index.replace("<OBJ", "").replace(">", ""))
        except ValueError:
            logger.warning(f"Invalid object index from caption: {repr(caption)}")
            return 0

    def _load(self):
        super()._load()
        # load predicted bboxes
        # self.input_predicted_bboxes = self.get_input_predicted_bbox()
        # self._compute_closest_predicted_bbox()

        if "train" in self.split:
            self._filter_data()


    def _filter_data(self, iou_threshold: float=0.75):
        logger.info(f"Filtering data that with IoU threshold {iou_threshold}...")
        logger.info(f"Before filtering: {len(self.annotation)}")
        self.annotation_old = self.annotation

        new_annotation = []
        for idx, item in enumerate(self.annotation):
            closet_pred_bbox = self.scene_data[item["scene_id"]]["closest_pred_bbox"][int(item["object_id"])]
            if closet_pred_bbox["iou"] >= iou_threshold:
                new_annotation.append(item)

        self.annotation = new_annotation
        logger.info(f"After filtering: {len(self.annotation)}, kept {len(self.annotation) / len(self.annotation_old) * 100:.2f}%")

    def _take_partial_data(self, ratio: float):
        # take different part for train and val
        # shuffle annotation
        random.seed(0)
        random.shuffle(self.annotation) # mix the data so that same scenes are used between train and val, but not same data
        if self.start_from_last:
            # take from last X percent
            self.annotation = self.annotation[-int(len(self.annotation) * ratio) :]
        else:
            self.annotation = self.annotation[: int(len(self.annotation) * ratio)]
            

    def get_annotation_file(self, split="train") -> str:
        DSET_PATH_SCANREFER = {
            "test": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_test.json",
            "train": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_train.json",
            "val": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_val.json",
        }
        # resplit
        DSET_PATH_SCANREFER_RESPLIT = {
            "train": f"{SVC_PATH}/scanrefer/ScanRefer_resplit_train.json",
            "val": f"{SVC_PATH}/scanrefer/ScanRefer_resplit_val.json",
        }
        DSET_PATH_SCANREFER_RESPLIT_SMALL = {
            "train": f"{SVC_PATH}/scanrefer/ScanRefer_resplit_small_train.json",
            "val": f"{SVC_PATH}/scanrefer/ScanRefer_resplit_small_val.json",
            "test": f"{SVC_PATH}/scanrefer/ScanRefer_resplit_small_train.json",
        }
        DSET_PATH_NR3D = {
            "train": f"{SVC_PATH}/Nr3D/nr3d_train.json",
            "val": f"{SVC_PATH}/Nr3D/nr3d_val.json",
            "test": f"{SVC_PATH}/Nr3D/nr3d_test.json",
        }
        DSET_PATH_SR3D = {
            "train": f"{SVC_PATH}/Sr3D/sr3d_train.json",
            "val": f"{SVC_PATH}/Sr3D/sr3d_val.json",
            "test": f"{SVC_PATH}/Sr3D/sr3d_test.json",
        }
        if "nr3d" in self.name:
            return DSET_PATH_NR3D[split]
        elif "sr3d" in self.name:
            return DSET_PATH_SR3D[split]
        elif "resplit-small" in self.name:
            return DSET_PATH_SCANREFER_RESPLIT_SMALL[split]
        elif "resplit" in self.name:
            return DSET_PATH_SCANREFER_RESPLIT[split]
        else:
            return DSET_PATH_SCANREFER[split]
    
    # def get_input_predicted_bbox(self) -> Dict[str, np.ndarray]:
    #     """
    #     assumed to return a scene_id -> list of predicted bboxes mapping
    #     no masked predicted bboxes!
    #     """
    #     if self.frozen_object_type == "pnpp-vote2cap-box":
    #         predicted_bbox_file = f"{SVC_PATH}/pc_features/scene_bbox_info_for_valtest_vote2cap_detr.pkl"
    #         logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")

    #         predicted_bbox = pickle.load(open(predicted_bbox_file, "rb"))
    #         # scene_id -> {bbox_id: {"bbox": ..., "is_valid": True/False}, ...}
    #         # NOTE: the predicted bboxes are already trimmed, so the index is the index among valid predicted bboxes
    #         for scene_id in predicted_bbox:
    #             # predicted_bbox[scene_id] = np.stack(
    #             #     [np.array(item["bbox"]) for item in predicted_bbox[scene_id] if item["is_valid"]]
    #             # )
    #             num_bboxes = len(predicted_bbox[scene_id])
    #             bboxes = np.zeros((num_bboxes, 6))
    #             for bbox_id, item in predicted_bbox[scene_id].items():
    #                 # print(item)
    #                 assert item["is_valid"]
    #                 if item["is_valid"]:
    #                     bboxes[bbox_id] = np.array(item["bbox"])

    #             predicted_bbox[scene_id] = bboxes

    #     elif self.frozen_object_type == "uni3d-mask3d-box":
    #         predicted_bbox_file = f"{SVC_PATH}/pc_features/chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt" # 1030d
    #         logger.info(f"Loading predicted bboxes from {predicted_bbox_file}...")
    #         predicted_bbox_anno = torch.load(predicted_bbox_file, map_location="cpu") # a list of dict

    #         # note that all of them are considered valid
    #         predicted_bbox = {}
    #         for item in predicted_bbox_anno:
    #             scene_id = item["scene_id"]
    #             bboxes = item["bbox"]
    #             predicted_bbox[scene_id] = bboxes.numpy() # [N_bboxes, 6]

    #         for scene_id in self.scene_list:
    #             if scene_id not in predicted_bbox:
    #                 logger.warning(f"Scene {scene_id} has no predicted bboxes!")
    #                 # predicted_bbox[scene_id] = np.zeros((100, 6))
    #                 predicted_bbox[scene_id] = np.random.rand(100, 6)

    #     else:
    #         raise NotImplementedError(f"Invalid frozen object type: {self.frozen_object_type} that have no predicted bboxes")
        
    #     return predicted_bbox

    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        # remove "." in description
        # instructions["description"] = instructions["description"].strip(".").strip()
        # remove "there/this/that/it is"
        # instructions["description"] = instructions["description"].replace("there is", "").replace("this is", "").replace("that is", "").replace("it is", "").strip()
        # modify object_index from gt box index to predicted box index (closest)
        if "description" in instructions and self.remove_bad_start_words:
            # remove "find" in "find xxxxxxx". this is mostly for Sr3D formatting
            # instructions["description"] = instructions["description"]
            bad_start_words = ["find", "select", "choose"]
            for bad_start_word in bad_start_words:
                if instructions["description"].startswith(bad_start_word):
                    instructions["description"] = instructions["description"][
                        len(bad_start_word) :
                    ].strip()
                    break
        
        
        object_index = instructions["object_index"] # GT object index
        new_object_index = self.scene_data[instructions["scene_id"]]["closest_pred_bbox"][object_index]["pred_id"]

        if (shuffled_indices_trimmed := kwargs.get("shuffled_indices_trimmed", None)) is not None:
            # shuffle object index according to shuffled object features
            new_object_index = shuffled_indices_trimmed[new_object_index]

        instructions["object_index"] = new_object_index
        return instructions

    # def _compute_closest_predicted_bbox(self):
    #     """
    #     calculate all IoU3D between GT bbox and predicted bboxes, find the closest one for each GT bbox
    #     """
    #     for scene_id in self.scene_list:
    #         pred_bboxes = self.input_predicted_bboxes[scene_id][:, :6]  # (K1, 6)
    #         # pred_bboxes_object_ids = self.input_predicted_bboxes[scene_id][:, -1]  # (K1,)
    #         pred_bboxes_object_ids = np.arange(len(pred_bboxes))  # (K1,)
    #         gt_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]  # (K2, 6)
    #         gt_object_ids = self.scene_data[scene_id]["instance_bboxes"][:, -1]  # (K2,)
            
            
    #         pred_idx_assigned, pred_iou = assign_preds_to_gts(pred_bboxes, gt_bboxes)  # (K2,), (K2,)
    #         for i, gt_object_id in enumerate(gt_object_ids):
    #             gt_object_id = int(gt_object_id)
    #             pred_id = int(pred_bboxes_object_ids[pred_idx_assigned[i]])
    #             iou = pred_iou[i]
                
    #             # Store the closest predicted bbox and its IoU for each GT bbox
    #             if "closest_pred_bbox" not in self.scene_data[scene_id]:
    #                 self.scene_data[scene_id]["closest_pred_bbox"] = {}
    #             self.scene_data[scene_id]["closest_pred_bbox"][gt_object_id] = {
    #                 "pred_id": pred_id,
    #                 "iou": iou
    #             }

    def evaluate(self, preds, gt_indices, iou_threshold=0.25) -> Tuple[str, Dict]:
        """
        preds: Map from scanrefer_id to predicted object index (index of input predicted bbox (ignored masked ones))
        gt_indices: Map from scanrefer_id to GT object index (in array, not real object id)

        returns: iou message and iou score
        """
        # correct, total = 0, 0
        correct_iou = 0
        ious = []
        common_keys = set(preds.keys()) & set(gt_indices.keys())
        logger.info(f"Common keys: {len(common_keys)}")
        logger.info(f"Total predictions: {len(preds)}")
        logger.info(f"Total GT: {len(gt_indices)}")
        if len(common_keys) != len(preds) or len(common_keys) != len(gt_indices):
            logger.warning("Some keys are missing in GT or predictions!")
        for scanrefer_id in common_keys:
            pred_caption = preds[scanrefer_id] 
            # shall be integer
            pred_id: int = self._parse_object_index(pred_caption) 
            gt_id: int = gt_indices[scanrefer_id]
            # scene_id, _, _ = scanrefer_id.split("|")
            scene_id = scanrefer_id.split("|")[1]

            assert scene_id in self.scene_list, f"Invalid scene ID: {scene_id}, current scene list: {self.scene_list}"

            gt_bbox = self.scene_data[scene_id]["instance_bboxes"][gt_id, :6]

            pred_bboxes = self.input_predicted_bboxes[scene_id]
            if pred_id >= len(pred_bboxes):
                logger.warning(f"Invalid predicted bbox index: {pred_id}, but only {len(pred_bboxes)} predicted bboxes")
                pred_id = 0

            pred_bbox = self.input_predicted_bboxes[scene_id][pred_id, :6]

            iou = box3d_iou_orthogonal(gt_bbox, pred_bbox)
            ious.append(iou)

            if iou >= iou_threshold:
                correct_iou += 1
        
        ious = np.array(ious)
        accuracy = correct_iou / len(common_keys)
        message = f"[Acc@{iou_threshold:.2f}] Mean: {np.mean(ious):.4f}, Max: {np.max(ious):.4f}, Min: {np.min(ious):.4f}, Acc: {correct_iou}/{len(common_keys)}={accuracy:.4f}"
        
        return message, {"accuracy": accuracy}


class Scan2CapSimpleDataset(VisualInstructionTuningDataset3D):
    """
    Reversed ScanRefer dataset, equals Scan2Cap and one object one instruction
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )
        # self.use_no_location_text = use_no_location_text
        # self.use_no_dataset_name = use_no_dataset_name

        self.use_no_location_text = kwargs.get("use_no_location_text")
        self.use_no_dataset_name = kwargs.get("use_no_dataset_name")
        # self.use_object_index = kwargs.get("use_object_index", False)

        # set ScanRefer-specific parameters
        # self.prompt = "" # TODO: refactor dataset annotation, to include target bbox in the annotation
        # if prompt is None:
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {description}" + f"{self.prompt_end_token}"
            postfix_name_desc = "\x04 {object_name}.\n {description}" + f"{self.prompt_end_token}"
            dataset_name = "ScanRefer"
            if "nr3d" in self.name:
                dataset_name = "Nr3D"
            elif "sr3d" in self.name:
                dataset_name = "Sr3D"
            logger.info(
                f"Using {dataset_name} style prompt for {self.name}-{self.split} dataset."
            )
            if self.use_no_dataset_name:
                self.prompt = [
                    f"Describe the object in detail at {{location}} in the room.\n",
                    f"Tell me about the object at {{location}} in the 3D scene.\n",
                    f"What is the object in the 3D scene at {{location}} ?\n",
                    f"How can I find the object at {{location}} in the 3D scene style?\n",
                    f"Could you detail the features of the object located at {{location}} within the 3D scene?\n",
                    f"What characteristics define the object at {{location}} in the room?\n" if dataset_name != "Sr3D" else "",
                    f"Can you describe the color and texture of the object at {{location}}?\n" if dataset_name != "Sr3D" else "",
                    f"In detail, how does the object at {{location}} stand out from its surroundings?\n",
                ]
            else:
                self.prompt = [
                    # "Describe the object at {location} in the 3D scene. ",
                    # "In the 3D scene, describe the object at {location}. ",
                    # "At xyzhwl {location}, describe the object in the 3D scene. ",
                    # "At xyzhwl {location}, give a description of the object. ",
                    # "The object at xyzhwl {location} in the 3D scene is: ",
                    # "The short name of object at {location} in the 3D scene is:\n",
                    # f"In {dataset_name} style, describe the object at {{location}} in the 3D scene.\n",
                    # f"In {dataset_name} style, describe the {{object_name}} at {{location}} in the 3D scene.\n"
                    f"In {dataset_name} style, describe the object in detail at {{location}} in the room.\n",
                    f"In {dataset_name} style, tell me about the object at {{location}} in the 3D scene.\n",
                    f"What is the object in the 3D scene at {{location}} in {dataset_name} style?\n",
                    f"How can I find the object at {{location}} in the 3D scene in {dataset_name} style?\n",
                    f"Could you detail the features of the object located at {{location}} within the 3D scene, employing the {dataset_name} dataset's approach?\n",
                    f"In the manner of the {dataset_name} dataset, what characteristics define the object at {{location}} in the room?\n" if dataset_name != "Sr3D" else "",
                    f"Can you describe the color and texture of the object at {{location}}, using {dataset_name} style?\n" if dataset_name != "Sr3D" else "",
                    f"In detail, how does the object at {{location}} stand out from its surroundings in the {dataset_name} dataset's interpretation?\n",
                    # f"What is the {{object_name}} in the 3D scene at {{location}} in {dataset_name} style?\n",
                    # f"In the 3D scene, describe in {dataset_name} style the object at {{location}}.\n",
                    # f"In {dataset_name} style, at xyzhwl {{location}}, describe the object in the 3D scene.\n",
                    # "Describe the object at {location} in the 3D scene.\n",
                    # "In the 3D scene, describe the object at {location}.\n",
                    # "At xyzhwl {location}, describe the object in the 3D scene.\n",
                ]

            

            self.prompt = [prompt for prompt in self.prompt if len(prompt) > 0]
            self.prompt = [prompt + postfix for prompt in self.prompt]

            self.prompt_name_desc = [
                f"In {dataset_name} style, what is the object's short name at {{location}} in the room? Describe the object in detail.\n",
                f"In {dataset_name} style, tell me about the short name of the object at {{location}} in the 3D scene and describe it.\n",
                f"What is the object short name and detailed description in the 3D scene at {{location}} in {dataset_name} style?\n",
            ]
            self.prompt_name_desc = [prompt + postfix_name_desc for prompt in self.prompt_name_desc]
            self.prompt.extend(self.prompt_name_desc)

            

            if self.use_no_location_text:
                logger.info(f"Using no-location text style for {self.get_dataset_description()}.")
                self.prompt = [prompt.replace("at {location}", "") for prompt in self.prompt]

        self.instruction_keys = ["description", "location", "object_name"]
        self.instruction_keys_for_prompt = ["location", "object_name"]

        if self.use_object_index:
            logger.info(f"Using object index text style for {self.get_dataset_description()}.")
            self.prompt = [
                p.replace("{location}", "{location} <OBJ{object_index}>") for p in self.prompt
            ]
            self.instruction_keys_for_prompt.append("object_index")
            self.instruction_keys.append("object_index")


        self.image_getter = get_scan2cap_image_for_instruction
        self.prepare_corpus()


    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        if "description" in instructions and self.remove_bad_start_words:
            # remove "find" in "find xxxxxxx"
            # instructions["description"] = instructions["description"]
            bad_start_words = ["find", "select", "choose"]
            for bad_start_word in bad_start_words:
                if instructions["description"].startswith(bad_start_word):
                    instructions["description"] = instructions["description"][
                        len(bad_start_word) :
                    ].strip()
                    break
        
        object_index = instructions["object_index"] # GT object index
        new_object_index = self.scene_data[instructions["scene_id"]]["closest_pred_bbox"][object_index]["pred_id"]

        if (shuffled_indices_trimmed := kwargs.get("shuffled_indices_trimmed", None)) is not None:
            # shuffle object index according to shuffled object features
            new_object_index = shuffled_indices_trimmed[new_object_index]

        instructions["object_index"] = new_object_index
        return instructions

    def get_annotation_file(self, split="train") -> str:
        DSET_PATH_SCANREFER = {
            "test": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_test.json",
            "train": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_train.json",
            "val": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_val.json",
        }
        DSET_PATH_NR3D = {
            "train": f"{SVC_PATH}/Nr3D/nr3d_train.json",
            "val": f"{SVC_PATH}/Nr3D/nr3d_val.json",
            "test": f"{SVC_PATH}/Nr3D/nr3d_test.json",
        }
        DSET_PATH_SR3D = {
            "train": f"{SVC_PATH}/Sr3D/sr3d_train.json",
            "val": f"{SVC_PATH}/Sr3D/sr3d_val.json",
            "test": f"{SVC_PATH}/Sr3D/sr3d_test.json",
        }
        if "nr3d" in self.name:
            return DSET_PATH_NR3D[split]
        elif "sr3d" in self.name:
            return DSET_PATH_SR3D[split]
        else:
            return DSET_PATH_SCANREFER[split]

    def prepare_corpus(self):
        logger.info("building corpus...")

        # helper function to prepare ground truth captions
        self.corpus = defaultdict(list)
        # object_id_to_name = defaultdict(lambda:'unknown')

        for data in self.annotation:

            # (         scene_id,         object_id,         object_name
            # ) = data["scene_id"], data["object_id"], data["object_name"]
            # key = f"{scene_id}|{object_id}|{object_name}"
            key = self._get_corpus_id(data)

            # parse language tokens
            # token = data["token"][:max_len]
            # description = " ".join(["sos"] + token + ["eos"])
            description = data["description"]
            description = preprocess_sos_eos_for_scan2cap(description)
            description = postprocess_punctuation_for_caption_metrics(description)
            # object_id_to_name[f"{scene_id}|{object_id}"] = object_name

            self.corpus[key].append(description)

        self.corpus = dict(self.corpus)

    def _get_corpus_id(self, data: Dict) -> str:
        (scene_id, object_id, object_name) = (
            data["scene_id"],
            data["object_id"],
            data["object_name"],
        )
        return f"{scene_id}|{object_id}|{object_name}"

    def _preprocess_annotation(self):
        # convert "instance_type" to "object_name"
        for data in self.annotation:
            if "instance_type" in data:
                data["object_name"] = data["instance_type"]
                data["object_id"] = data["target_id"]

    def deduplicate_captions(self):
        """
        Deduplicate annotation for training - for each object id, keep only one caption.
        """
        logger.info(f"Deduplicating annotation for {self.get_dataset_description()}...")
        dedup_annotation = []
        seen = set()
        for data in self.annotation:
            key = self._get_corpus_id(data)
            if key not in seen:
                seen.add(key)
                dedup_annotation.append(data)

        self.annotation = dedup_annotation
        logger.info(f"Deduplicated to {len(self.annotation)} samples.")



class Scan2CapTestDataset(Scan2CapSimpleDataset):
    """
    Test dataset for Scan2Cap, load object from predicted results.
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        self.bbox_file = kwargs.get("bbox_file")
        self.add_dummy_caption = kwargs.get("add_dummy_caption", True)
        self.use_gt_bbox = kwargs.get("use_gt_bbox")

        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # if prompt is None:
        if kwargs.get("prompt", None) is None:
            # Use a single-template for test
            postfix = "\x04 {description}" + f"{self.prompt_end_token}"
            dataset_name = "ScanRefer"
            if "nr3d" in self.name:
                dataset_name = "Nr3D"
            elif "sr3d" in self.name:
                dataset_name = "Sr3D"
            logger.info(
                f"Using {dataset_name} style prompt for {self.name}-{self.split} dataset."
            )

            if self.use_no_dataset_name:
                self.prompt = [
                    f"Describe the object in detail at {{location}} in the room.\n",
                ]
            else:
                self.prompt = [
                    # "Describe the object at {location} in the 3D scene. ",
                    # "In the 3D scene, describe the object at {location}. ",
                    # "At xyzhwl {location}, describe the object in the 3D scene. ",
                    # "At xyzhwl {location}, give a description of the object. ",
                    # "The object at xyzhwl {location} in the 3D scene is: ",
                    # "The short name of object at {location} in the 3D scene is:\n",
                    f"In {dataset_name} style, describe the object in detail at {{location}} in the room.\n",
                    # f"In the 3D scene, describe in {dataset_name} style the object at {{location}}.\n",
                    # f"In {dataset_name} style, at xyzhwl {{location}}, describe the object in the 3D scene.\n",
                    # "Describe the object at {location} in the 3D scene.\n",
                ]
            self.prompt = [prompt + postfix for prompt in self.prompt]


            if self.use_no_location_text:
                logger.info(f"Using no-location text style for {self.get_dataset_description()}.")
                self.prompt = [prompt.replace("at {location}", "") for prompt in self.prompt]

        if self.use_object_index:
            logger.info(f"Using object index text style for {self.get_dataset_description()}.")
            self.prompt = [p.replace("{location}", "{location} <OBJ{object_index}>") for p in self.prompt]
            self.instruction_keys_for_prompt.append("object_index")
            self.instruction_keys.append("object_index")

        assert split in ["test", "val"]
        if not self.use_gt_bbox:
            self._build_new_annotation()

    # load bbox from predicted results
    def _load(self):
        """
        Load 3D scene, instance and object information
        """
        if self.use_gt_bbox:
            super()._load()
            return

        logger.info("Loading scene (ScanNet) data...")
        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.annotation])))
        logger.info(self.name)
        logger.info(self.scene_list)
        # self.scene_list = self.get_all_scene_ids()

        # load predicted bbox
        all_bbox_data = pickle.load(open(self.bbox_file, "rb"))  # axis-aligned

        # load scene data
        self.scene_data = {}
        scene_path = self.get_scene_path()
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_vert.npy"
            )  # axis-aligned
            bbox_data = all_bbox_data[scene_id]  # {bbox_id : {"bbox": bbox, ...}}
            bboxes = np.zeros((len(bbox_data), 7), dtype=np.float32)
            for bbox_id, bbox in bbox_data.items():
                bboxes[int(bbox_id)][:6] = bbox["bbox"]
                bboxes[int(bbox_id)][-1] = int(bbox_id)

            self.scene_data[scene_id]["instance_bboxes"] = bboxes
            self.scene_data[scene_id]["instance_bboxes_gt"] = np.load(
                os.path.join(scene_path, scene_id) + "_aligned_bbox.npy"
            )

            # dummy instance labels and semantic labels
            self.scene_data[scene_id]["semantic_labels"] = np.zeros(
                (self.scene_data[scene_id]["mesh_vertices"].shape[0],), dtype=np.int32
            )
            self.scene_data[scene_id]["instance_labels"] = np.zeros(
                (self.scene_data[scene_id]["mesh_vertices"].shape[0],), dtype=np.int32
            )
            try:
                axis_align_matrix = json.load(open("data/alignments.json", "r"))[scene_id]
                axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
            except KeyError:
                axis_align_matrix = np.eye(4) # for test scenes
            self.scene_data[scene_id]["axis_align_matrix"] = axis_align_matrix

        self.input_predicted_bboxes = self.get_input_predicted_bbox()
        self._compute_closest_predicted_bbox()

    def _build_new_annotation(self):
        """
        build new annotation for inference (with predicted bbox)
        """
        self.annotation_gt = deepcopy(self.annotation)
        self.annotation = []
        for scene_id in self.scene_list:
            for object_id, bbox in enumerate(self.scene_data[scene_id]["instance_bboxes"]):
                self.annotation.append(
                    {
                        "scene_id": scene_id,
                        "object_id": object_id,
                        "object_name": "unknown",
                        "description": "unknown",
                    }
                )

    def process_predictions(
        self,
        predictions: Dict[str, str],
        iou_threshold: float = 0.5,
        dummy_caption: str = "sos eos",
        method: str = "recall",
    ):
        """
        Process predictions to get caption corpus, organized by scene_id -> object_id (gt) -> caption
        """
        # get gt corpus
        gt_scene_id_to_object_id_to_caption = defaultdict(lambda: defaultdict(set))
        pred_scene_id_to_object_id_to_caption = defaultdict(dict)
        # used_gt_object_ids = defaultdict(set)
        for key, captions in self.corpus.items():
            scene_id, object_id, object_name = key.split("|")
            gt_scene_id_to_object_id_to_caption[scene_id][object_id].update(captions)

        # convert set to list
        for (
            scene_id,
            object_id_to_caption,
        ) in gt_scene_id_to_object_id_to_caption.items():
            for object_id, captions in object_id_to_caption.items():
                gt_scene_id_to_object_id_to_caption[scene_id][object_id] = list(captions)

        # print(gt_scene_id_to_object_id_to_caption)
        # organize predictions
        predictions_corpus = defaultdict(dict)
        for key, caption in predictions.items():
            # print(key)
            scene_id, object_id, object_name = key.split(
                "|"
            )  # object_id is the predicted object_id
            predictions_corpus[scene_id][object_id] = caption
                
        if method == "recall":
            logger.info("Calculating recall...")
            # assign predictions to gt
            accepted_iou, used_gts = 0, 0
            for scene_id in predictions_corpus.keys():
                pred_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]  # (K1, 6)
                pred_bboxes_object_ids = self.scene_data[scene_id]["instance_bboxes"][
                    :, -1
                ]  # (K1,)
                gt_bboxes = self.scene_data[scene_id]["instance_bboxes_gt"][:, :6]  # (K2, 6)
                gt_object_ids = self.scene_data[scene_id]["instance_bboxes_gt"][:, -1]  # (K2,)
                pred_idx_assigned, pred_iou = assign_preds_to_gts(
                    pred_bboxes, gt_bboxes
                )  # (K2,), (K2,)
                
                # each gt bbox is assigned to a pred bbox
                # then, each pred bbox is assigned to a gt bbox, if iou > iou_threshold
                for i, gt_object_id in enumerate(gt_object_ids):
                    gt_object_id = str(int(gt_object_id))
                    # print(scene_id, gt_object_id)
                    if gt_object_id not in gt_scene_id_to_object_id_to_caption[scene_id]:
                        # print("ignoring", scene_id, ",", gt_object_id)
                        continue  # skip no gt object

                    used_gts += 1
                    # pred_id = pred_idx_assigned[i]
                    pred_id = int(
                        pred_bboxes_object_ids[pred_idx_assigned[i]]
                    )  # for non-gt bbox, pred_bboxes_object_ids is simply a range starting from 0
                    if pred_id >= 0 and pred_iou[i] >= iou_threshold:
                        assert (
                            gt_object_id not in pred_scene_id_to_object_id_to_caption[scene_id]
                        )  # should not have duplicate object_id
                        pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id] = (
                            predictions_corpus[scene_id][str(pred_id)]
                        )
                        accepted_iou += 1

            accepted_rate = accepted_iou / used_gts if used_gts > 0 else 0
            logger.info(
                f"Accepted bbox@{iou_threshold}: {accepted_iou}/{used_gts}={accepted_rate:.4f}"
            )

            pred = self._flatten_corpus(pred_scene_id_to_object_id_to_caption)
            gt = self._flatten_corpus(gt_scene_id_to_object_id_to_caption)

            # show some examples
            if len(pred) > 0 and len(gt) > 0:
                common_keys = list(set(pred.keys()).intersection(set(gt.keys())))
                show_keys = np.random.choice(common_keys, N_SHOW_CAPTION_SAMPLES)
                logger.info(f"Showing {N_SHOW_CAPTION_SAMPLES} examples:")
                for key in show_keys:
                    logger.info(f"GT: {key}: {gt[key]}")
                    logger.info(f"Pred: {key}: {pred[key]}")

            # add unmatched gt
            if self.add_dummy_caption:
                self._add_missing_gt(gt, pred, dummy_caption=dummy_caption)

            # remove unmatched pred
            pred = self._organize_corpus(gt, pred)

            return gt, pred
        
        elif method == "precision":
            # assign gt to predictions
            accepted_iou, used_preds = 0, 0
            for scene_id in gt_scene_id_to_object_id_to_caption.keys():
                pred_bboxes = self.scene_data[scene_id]["instance_bboxes"][:, :6]
                pred_bboxes_object_ids = self.scene_data[scene_id]["instance_bboxes"][:, -1] 
                gt_bboxes = self.scene_data[scene_id]["instance_bboxes_gt"][:, :6]  # (K2, 6)
                gt_object_ids = self.scene_data[scene_id]["instance_bboxes_gt"][:, -1]  # (K2,)
                gt_idx_assigned, gt_iou = assign_gts_to_preds(
                    pred_bboxes, gt_bboxes
                )  # (K1,), (K1,)

                # each pred bbox is assigned to a gt bbox
                # then, each gt bbox is assigned to a pred bbox, if iou > iou_threshold
                # good_assign_mask = gt_iou >= iou_threshold # (K1,)
                for i, pred_object_id in enumerate(pred_bboxes_object_ids):
                    gt_object_id = str(int(gt_object_ids[gt_idx_assigned[i]]))
                    if gt_object_id not in gt_scene_id_to_object_id_to_caption[scene_id]:
                        # print("ignoring", scene_id, ",", gt_object_id)
                        continue # skip no gt object

                    used_preds += 1
                    if gt_iou[i] >= iou_threshold:
                        pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id] = (
                            predictions_corpus[scene_id][str(int(pred_object_id))] # normally, pred_object_id is the same as i
                        )
                        accepted_iou += 1
                    else:
                        # add unmatched pred
                        pred_scene_id_to_object_id_to_caption[scene_id][gt_object_id] = [dummy_caption]
                
            accepted_rate = accepted_iou / used_preds if used_preds > 0 else 0
            logger.info(
                f"Accepted bbox@{iou_threshold}: {accepted_iou}/{used_preds}={accepted_rate:.4f}"
            )
            pred = self._flatten_corpus(pred_scene_id_to_object_id_to_caption)
            gt = self._flatten_corpus(gt_scene_id_to_object_id_to_caption)
                    
            # show some examples
            if len(pred) > 0 and len(gt) > 0:
                common_keys = list(set(pred.keys()).intersection(set(gt.keys())))
                show_keys = np.random.choice(common_keys, N_SHOW_CAPTION_SAMPLES)
                logger.info(f"Showing {N_SHOW_CAPTION_SAMPLES} examples:")
                for key in show_keys:
                    logger.info(f"GT: {key}: {gt[key]}")
                    logger.info(f"Pred: {key}: {pred[key]}")

            # remove unmatched gt
            gt = self._organize_corpus(pred, gt)

            return gt, pred
        
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

    def _add_missing_gt(self, gt, pred, dummy_caption="sos eos"):
        for k in gt.keys():
            if k not in pred:
                pred[k] = [dummy_caption]

    def _organize_corpus(self, gt, pred):
        # remove unmatched pred
        new_pred = {}
        for k in gt.keys():
            new_pred[k] = pred[k]

        return new_pred

    def _flatten_corpus(self, corpus: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """
        Flatten corpus to a list of captions
        """
        new_corpus = {}
        for scene_id, object_id_to_caption in corpus.items():
            for object_id, captions in object_id_to_caption.items():
                new_corpus[f"{scene_id}|{object_id}"] = captions

        return new_corpus


class Scan2ObjectNameDataset(Scan2CapSimpleDataset):
    """
    Output object names instead of descriptions
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # set ScanRefer-specific parameters
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {object_name}" + f"{self.prompt_end_token}"
            self.prompt = [
                # "Describe the object at {location} in the 3D scene. ",
                # "In the 3D scene, describe the object at {location}. ",
                # "At xyzhwl {location}, describe the object in the 3D scene. ",
                # "At xyzhwl {location}, give a description of the object. ",
                # "The object at xyzhwl {location} in the 3D scene is: ",
                "The short name of object at {location} in the room is:\n",
                "Describe the the short name of object at {location} in the room.\n",
                "Give the short name of object at {location} in the room:\n",
                "What's the short name for the object at {location}?\n",
                "Identify the object at {location} with a brief name.\n",
                "What is the object at {location} commonly called in a nutshell?\n",
            ]
            self.prompt = [prompt + postfix for prompt in self.prompt]
            
            
            if self.use_no_location_text:
                logger.info(f"Using no-location text style for {self.get_dataset_description()}.")
                self.prompt = [prompt.replace("at {location}", "") for prompt in self.prompt]

        self.instruction_keys = ["object_name", "location"]
        self.instruction_keys_for_prompt = ["location"]

        if self.use_object_index:
            logger.info(f"Using object index in prompt for {self.get_dataset_description()}.")
            self.prompt = [p.replace("{location}", "{location} <OBJ{object_index}>") for p in self.prompt]
            self.instruction_keys.append("object_index")
            self.instruction_keys_for_prompt.append("object_index")

    def get_annotation_file(self, split="train") -> str:
        if split == "merged":
            return self.load_merged_annotation() # load SCANREFER, NR3D, SR3D merged annotation
        else:
            return super().get_annotation_file(split)

    def load_merged_annotation(self):
        DSET_PATH_SCANREFER = {
            "test": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_test.json",
            "train": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_train.json",
            "val": f"{SVC_PATH}/scanrefer/ScanRefer_filtered_val.json",
        }
        DSET_PATH_NR3D = {
            "train": f"{SVC_PATH}/Nr3D/nr3d_train.json",
            "val": f"{SVC_PATH}/Nr3D/nr3d_val.json",
            "test": f"{SVC_PATH}/Nr3D/nr3d_test.json",
        }
        DSET_PATH_SR3D = {
            "train": f"{SVC_PATH}/Sr3D/sr3d_train.json",
            "val": f"{SVC_PATH}/Sr3D/sr3d_val.json",
            "test": f"{SVC_PATH}/Sr3D/sr3d_test.json",
        }

        # load train annotations
        annotations = []
        for DSET_KIND in [DSET_PATH_SCANREFER, DSET_PATH_NR3D, DSET_PATH_SR3D]:
            annotation_file = DSET_KIND["train"]
            with open(annotation_file, "r") as f:
                annotations.extend(json.load(f))

        return annotations


class ScanQADatasetUnified(VisualInstructionTuningDataset3D):
    """
    Unified dataset for ScanQA and SQA3D
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # set ScanRefer-specific parameters
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {answer}" + f"{self.prompt_end_token}"
            if "scanqa-mv" in self.name:
                self.prompt = [
                    "Answer the following Multi-view ScanQA question based on the image and the room:{question}\n",
                ]
            elif "scanqa" in self.name:
                self.prompt = [
                    "Answer the following ScanQA question based on the image and the room:{question}\n",
                ]
            elif "sqa3d" in self.name:
                self.prompt = [
                    "Currently my situation: {situation} Answer the following SQA3D question based on the situation, image and the room:{question}\n",
                ]
            else:
                # raise NotImplementedError(f"Unknown dataset name: {self.name}")
                self.prompt = [
                    f"Answer the following {self.name.upper()} question based on the image and the room:{{question}}\n",
                ]

            self.prompt = [prompt + postfix for prompt in self.prompt]

        self.instruction_keys = ["question", "answers", "situation"]
        self.instruction_keys_for_prompt = ["question", "situation"]
        self.image_getter = kwargs.get("image_getter", None)
        if self.image_getter is None:
            if "scanqa-mv" in self.name:
                self.image_getter = get_scanqa_mv_images_for_question
            elif "scanqa" in self.name:
                self.image_getter = get_scanqa_image_for_question
            elif "sqa3d" in self.name:
                self.image_getter = get_sqa3d_image_for_question
            elif "lamm3d" in self.name or "scenecap" in self.name:
                self.image_getter = dummy_image_getter
    
        self.build_answer_vocab()

    
    def get_difficulty_split_map(self):
        assert "scanqa-mv" in self.name, "Difficulty split map is only available for ScanQA-MV"
        difficulties = set([data["n_views_can_solve"] for data in self.annotation])
        print_once(f"{self.get_dataset_description()} has {len(difficulties)} difficulty levels: {difficulties}")
        difficulty_split_map = defaultdict(list)
        for anno in self.annotation:
            difficulty = anno["n_views_can_solve"]
            difficulty_split_map[difficulty].append(anno)

        return difficulty_split_map

    def get_question_id_to_difficulty(self):
        assert "scanqa-mv" in self.name, "Difficulty split map is only available for ScanQA-MV"
        question_id_to_difficulty = {}
        for anno in self.annotation:
            question_id = anno["question_id"]
            difficulty = anno["n_views_can_solve"]
            question_id_to_difficulty[question_id] = difficulty

        return question_id_to_difficulty


    def get_annotation_file(self, split="train") -> str:
        DSET_PATH_SCANQA = {
            "test_w_obj": f"{SVC_PATH}/ScanQA_v1.0_test_w_obj.json",
            "test_wo_obj": f"{SVC_PATH}/ScanQA_v1.0_test_wo_obj.json",
            "train": f"{SVC_PATH}/ScanQA_v1.0_train.json",
            "val": f"{SVC_PATH}/ScanQA_v1.0_val.json",
        }
        DSET_PATH_SQA3D = {
            "test": f"{SVC_PATH}/SQA_test.json",
            "train": f"{SVC_PATH}/SQA_train.json",
            "val": f"{SVC_PATH}/SQA_val.json",
        }
        DSET_PATH_SCANQA_MV = {
            "train": f"{SVC_PATH}/qa/ScanQA_mv_train_filtered_cleaned.json",
            "val": f"{SVC_PATH}/qa/ScanQA_mv_val_filtered_cleaned.json",
        }

        if "scanqa-mv" in self.name:
            return DSET_PATH_SCANQA_MV[split]
        elif "scanqa" in self.name:
            return DSET_PATH_SCANQA[split]
        elif "sqa" in self.name:
            return DSET_PATH_SQA3D[split]
        else:
            raise NotImplementedError(f"Unknown dataset name: {self.name}")

    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        # sample one answer
        if isinstance(instructions["answers"], list) or isinstance(instructions["answers"], tuple):
            instructions["answer"] = random.choice(instructions["answers"])
        elif isinstance(instructions["answers"], str):
            instructions["answer"] = instructions["answers"]
        return instructions

    def build_answer_vocab(self):
        """
        Build answer vocabulary list for generating answers
        """
        self.answer_vocab = set()
        for data in self.annotation:
            if "answers" in data:
                answers = data["answers"]
                self.answer_vocab.update(answers)

        logger.info(f"Answer vocab size of {self.name}-{self.split}: {len(self.answer_vocab)}")


class OpenEndedQADataset(ScanQADatasetUnified):
    def get_annotation_file(self, split="train") -> str:
        DSET_LAMM_3D_INSTRUCTION = {
            "train": f"{SVC_PATH}/lamm/3D_Instruct_meta_file_VQA_ScanQA_multiplechoice_finetune.json",
        }
        return DSET_LAMM_3D_INSTRUCTION[split]

    def _preprocess_annotation(self):
        self._old_annotation = deepcopy(self.annotation)
        self.annotation = []

        for data in self._old_annotation:
            self.annotation.append(
                {
                    "scene_id": data["id"],
                    "question": data["conversations"][0]["value"]
                    .replace("\n", " ")
                    .replace("Context: N/A", "")
                    .strip(),
                    "answers": [
                        data["conversations"][1]["value"].replace("\n###\n", " ").strip()
                    ],
                }
            )

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )
        self.image_getter = dummy_image_getter
        # if prompt is None:
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {answer}" + f"{self.prompt_end_token}"
            self.prompt = [
                "Answer the following multiple-choice question based on the room: {question}\n",
            ]
            self.prompt = [prompt + postfix for prompt in self.prompt]


class SceneCaptionDataset(VisualInstructionTuningDataset3D):
    """
    Scene caption dataset
    `name` should be "scenecap"
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # set ScanRefer-specific parameters
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {description}" + f"{self.prompt_end_token}"
            self.prompt = [
                "Describe the 3D scene.\n",
                "Describe the room.\n ",
                "Introduce the 3D scene.\n",
            ]
            self.prompt = [prompt + postfix for prompt in self.prompt]

        self.instruction_keys = ["answers"]
        self.instruction_keys_for_prompt = []
        if self.image_getter is None:
            self.image_getter = dummy_image_getter

    def get_annotation_file(self, split="train") -> str:
        DSET_SCENECAP = {
            "train": f"{SVC_PATH}/3dllm/3d_llm_scene_description_train.json",
            "val": f"{SVC_PATH}/3dllm/3d_llm_scene_description_val.json",
        }

        return DSET_SCENECAP[split]

    def process_instructions(self, instructions: Dict, **kwargs) -> Dict:
        instructions["description"] = random.choice(instructions["answers"])
        instructions["description"] = (
            instructions["description"].replace("<box>", "box").replace("<point>", "point")
        )
        return instructions

class ScanNetFrameCaptionDataset(VisualInstructionTuningDataset3D):
    """
    Frame caption dataset
    `name` should be "framecap"
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        self.percentile = kwargs.get("percentile")
        self.dataset_type = kwargs.get("dataset_type")
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # set ScanRefer-specific parameters
        # self.prompt = "" # TODO: refactor dataset annotation, to include target bbox in the annotation
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {description}" + f"{self.prompt_end_token}"
            self.prompt = [
                "Describe the 3D scene in the image.\n",
                "Describe the room about the given objects.\n",
                "Tell me about the 3D scene in the image.\n",
                "What can I see in the room in this view?\n",
            ]
            self.prompt = [prompt + postfix for prompt in self.prompt]

        self.instruction_keys = ["description"]
        self.instruction_keys_for_prompt = []
        if self.image_getter is None:
            self.image_getter = get_frame2cap_image_for_instruction

    def get_annotation_file(self, split="train") -> str:
        # percentile = 30.0
        if len(self.percentile) > 0:
            postfix = f"_with_score_{self.percentile}"
        else:
            postfix = "_with_score"

        DSET_FRAMECAP_PATHS = {
            "framecap": {
                "train": f"{SVC_PATH}/annotations{postfix}_train.json",
                "val": f"{SVC_PATH}/annotations{postfix}_val.json",
                "test": f"{SVC_PATH}/annotations{postfix}_test.json",
            },
            "framecap-2": {
                "train": f"{SVC_PATH}/annotations{postfix}_train.json",
                "val": f"{SVC_PATH}/annotations{postfix}_val.json",
                "test": f"{SVC_PATH}/annotations{postfix}_test.json",
            },
            "framecap-gpt4o": {
                "train": f"{SVC_PATH}/api-captions/annotations_gpt4o_train.json",
                "val": f"{SVC_PATH}/api-captions/annotations_gpt4o_val.json",
                "test": f"{SVC_PATH}/api-captions/annotations_gpt4o_test.json",
            }
        }


        return DSET_FRAMECAP_PATHS[self.dataset_type][split]

class ScanNetFrameQADataset(VisualInstructionTuningDataset3D):
    """
    Frame caption dataset
    `name` should be "frameqa"
    """

    def __init__(
        self,
        name: str,
        split: str,
        ratio: float,
        **kwargs,
    ):
        # self.percentile = kwargs.get("percentile")
        self.dataset_type = kwargs.get("dataset_type")
        super().__init__(
            name,
            split,
            ratio,
            **kwargs,
        )

        # set ScanRefer-specific parameters
        # self.prompt = "" # TODO: refactor dataset annotation, to include target bbox in the annotation
        if kwargs.get("prompt", None) is None:
            postfix = "\x04 {answer}" + f"{self.prompt_end_token}"
            self.prompt = [
                "Answer the following FrameQA question based on the image and the room:{question}\n",
                "What is the short answer to the FrameQA question based on the image and the room:{question}\n",
                "FrameQA quetion: {question}\n",
                # "Describe the 3D scene in the image.\n",
                # "Describe the room about the given objects.\n",
                # "Tell me about the 3D scene in the image.\n",
                # "What can I see in the room in this view?\n",
            ]
            self.prompt = [prompt + postfix for prompt in self.prompt]

        self.instruction_keys = ["question", "answer"]
        self.instruction_keys_for_prompt = []
        if self.image_getter is None:
            self.image_getter = get_frame2cap_image_for_instruction

    def get_annotation_file(self, split="train") -> str:
        DSET_FRAMEQA_PATHS = {
            "frameqa": {
                "train": f"{SVC_PATH}/frameqa_70.0_train.json",
                "val": f"{SVC_PATH}/frameqa_70.0_val.json",
                "test": f"{SVC_PATH}/frameqa_70.0_test.json",
            }
        }

        return DSET_FRAMEQA_PATHS[self.dataset_type][split]
    

def batch_parse_grounding_text(predictions: List[str]) -> List[List[float]]:
    """
    Parse grounding output to get the target bbox
    """
    # get target bbox location, if needed
    # predictions = [m.split('\x04', 1)[1].strip() if '\x04' in m else m for m in predictions] # [x, y, z, h, w, l]
    # predictions = [m.strip("[]").split(",") for m in predictions]
    results = []
    for pred in predictions:
        try:
            # if "\x04" in pred:
            # pred = pred.split("\x04")[1].strip()
            # pred = pred.strip("[]").split(",")
            # pred = pred.split(" ")
            # pred = [p.strip("<>") for p in pred]
            # pred = [float(p.strip()) / SCALE_DETECTION for p in pred]
            # pred = pred.split("<box>")[1].split("</box>")[0].split(") (")
            # pred = [p.split(",") for p in pred]

            # pred = [p.strip("<>") for p in pred]
            # pred = parse("({},{},{}) ({},{},{})", pred)
            pred = parse("[{},{},{},{},{},{}]", pred)
            assert pred is not None, "bbox should be [x, y, z, h, w, l]"
            # assert len(pred) == 6, "bbox should be [x, y, z, h, w, l]"
            # assert len(pred.fixed) == 6, "bbox should be [x, y, z, h, w, l]"

            pred = [float(p.strip()) / SCALE_DETECTION for p in pred]
            # pred = from_minmax_to_xyzhwl(np.array(pred))

        except Exception as e:
            print(f"Not found bbox in {pred}")
            # print(e)
            # pred = [0, 0, 0, 0, 0, 0]
            # pred = [-100, -100, -100, 1, 1, 1] # give a dummy bbox
            pred = INVALID_BBOX
        results.append(pred)
    # print INVALID ratio for debugging
    invalid_ratio = sum([1 for r in results if r == INVALID_BBOX]) / len(results)
    print(f"Invalid bbox ratio: {invalid_ratio}")
    return results


def get_iou(prediction, target):
    """
    prediction: [x, y, z, h, w, l]
    target: [x, y, z, h, w, l]
    """
    prediction_iou = get_3d_box(prediction[3:], 0, prediction[:3])
    target_iou = get_3d_box(target[3:], 0, target[:3])
    try:
        iou_3D, iou_2D = box3d_iou(prediction_iou, target_iou)
    except Exception as e:
        print(f"Error in calculating iou: {e}")
        iou_3D, iou_2D = 0, 0
    return iou_3D

# not used
def batch_parse_get_iou(preds: List[str], targets: List[List[float]]):
    """
    preds: parsed to be [N, 6]
    targets: [N, 6]
    """
    preds = batch_parse_grounding_text(preds)
    # targets = batch_parse_grounding_text(targets)
    preds = np.array(preds)
    targets = np.array(targets)
    print(preds, targets)
    ious = []
    for pred, target in zip(preds, targets):
        iou = get_iou(pred, target)
        ious.append(iou)
    ious = np.array(ious)
    print(ious)
    return ious


# not used
def acc_iou(preds: List[str], targets: List[List[float]], threshold: float):
    """
    preds: parsed to be [N, 6]
    targets: [N, 6]
    """
    preds = batch_parse_grounding_text(preds)
    # targets = batch_parse_grounding_text(targets)
    preds = np.array(preds)
    targets = np.array(targets)
    ious = []
    for pred, target in zip(preds, targets):
        iou = get_iou(pred, target)
        ious.append(iou)
    ious = np.array(ious)
    return np.sum(ious > threshold) / len(ious)


def calculate_reinforce_reward_labels(
    target: torch.Tensor, vocab: Dict[str, int], sigma: float = 1
):
    """
    Calculate the REINFORCE reward for a target integer.
    Reward is simple MLE
    """
    # monkey patch:
    distances = (torch.arange(0, 1000).to(target.device) + vocab["0"] - target) ** 2 / sigma**2
    rewards = torch.exp(
        -distances
    )  # r = exp(-d^2 / sigma^2), d = (i - target), i = 0, 1, ..., 999 || means the reward is 1 when i == target

    reward_labels = torch.zeros(len(vocab)).to(target.device)
    digit_vocab = [vocab[str(i)] for i in range(1000)]
    reward_labels[digit_vocab] = rewards
    return reward_labels


def batch_calculate_reinforce_reward_labels(
    targets: torch.LongTensor, vocab: Dict[str, int], sigma: float = 1
):
    """
    Calculate the REINFORCE reward for a batch of target integers.
    targets: [N]
    """
    reward_labels_whole = torch.zeros((len(targets), len(vocab)), device=targets.device)
    for i, target in enumerate(targets):
        if target < 0:  # ignore token
            reward_labels = torch.zeros(len(vocab)).to(targets.device)
        elif target >= vocab["0"] and target <= vocab["999"]:  # digit token
            reward_labels = calculate_reinforce_reward_labels(target, vocab, sigma)
            # NOTE: debug - this shall be same as pure CE loss
            # reward_labels = torch.zeros(len(vocab)).to(targets.device)
            # reward_labels[target] = 1
        else:  # other (text) token
            reward_labels = torch.zeros(len(vocab)).to(targets.device)
            reward_labels[target] = 1

        reward_labels_whole[i] = reward_labels

    return reward_labels_whole


def score_captions(corpus: dict, candidates: dict):
    """
    adapted from Vote2Cap-DETR
    """

    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    try:
        meteor = capmeteor.Meteor().compute_score(corpus, candidates)
    except Exception as e:
        logger.warning("METEOR failed:")
        print(e)
        meteor = deepcopy(rouge)


    score_per_caption = {
        "bleu-1": [float(s) for s in bleu[1][0]],
        "bleu-2": [float(s) for s in bleu[1][1]],
        "bleu-3": [float(s) for s in bleu[1][2]],
        "bleu-4": [float(s) for s in bleu[1][3]],
        "cider": [float(s) for s in cider[1]],
        "rouge": [float(s) for s in rouge[1]],
        "meteor": [float(s) for s in meteor[1]],
    }

    message = "\n".join(
        [
            "[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][0], max(bleu[1][0]), min(bleu[1][0])
            ),
            "[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][1], max(bleu[1][1]), min(bleu[1][1])
            ),
            "[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][2], max(bleu[1][2]), min(bleu[1][2])
            ),
            "[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                bleu[0][3], max(bleu[1][3]), min(bleu[1][3])
            ),
            "[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                cider[0], max(cider[1]), min(cider[1])
            ),
            "[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                rouge[0], max(rouge[1]), min(rouge[1])
            ),
            "[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(
                meteor[0], max(meteor[1]), min(meteor[1])
            ),
        ]
    )

    eval_metric = {
        "BLEU-4": bleu[0][3],
        "CiDEr": cider[0],
        "Rouge": rouge[0],
        "METEOR": meteor[0],
    }
    return score_per_caption, message, eval_metric


def metrics_qa(preds: str, labels: list[str]):
    em = 0
    if preds in labels:
        em = 1

    refined_em = 0
    for label in labels:
        if "".join(label.split()) in "".join(preds.split()):
            refined_em = 1
            break
        if "".join(preds.split()) in "".join(label.split()):
            refined_em = 1
            break

    return em, refined_em


def compute_qa_score(preds: Dict[str, str], labels: Dict[str, List[str]]):
    """
    preds: {"question_id": "answer" or ["answer"]}
    labels: {"question_id": ["answer1", "answer2"]}
    """
    em = 0
    refined_em = 0
    for question_id, pred in preds.items():
        if isinstance(pred, list):
            pred = pred[0]
        em_, refined_em_ = metrics_qa(pred, labels[question_id])
        em += em_
        refined_em += refined_em_
    return em / len(preds), refined_em / len(preds)


def loss_reinforce(logits, reward_labels):
    """
    logits: [B, N, V] or [..., V]
    reward_labels: [B, N, V] or [..., V]
    """
    # loss = None
    # if labels is not None:
    #     # Shift so that tokens < n predict n
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     # Flatten the tokens
    #     loss_fct = CrossEntropyLoss()
    #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
    #     shift_labels = shift_labels.view(-1)
    #     # Enable model parallelism
    #     shift_labels = shift_labels.to(shift_logits.device)
    #     loss = loss_fct(shift_logits, shift_labels)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_reward_labels = reward_labels[..., 1:, :].contiguous()
    # loss = -log_softmax(shift_logits, dim=-1) * shift_reward_labels
    loss = -torch.log_softmax(shift_logits, dim=-1) * shift_reward_labels  # [B, N, V]
    loss = loss.sum(-1).mean()
    return loss


class MergedDataset(Dataset, LogDatasetMixin):
    """
    wrapper for multiple datasets.
    get from each dataset by index
    """

    def __init__(
        self,
        datasets: List[VisualInstructionTuningDataset3D],
        sample_with_sqrt_freq: bool = False,
        annealing_schedule: list[tuple[float, float]] = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.seed = seed
        self.shuffle = shuffle

        # sample with sqrt frequency
        self.sample_with_sqrt_freq = sample_with_sqrt_freq
        if sample_with_sqrt_freq:
            logger.info("Sample with sqrt frequency")
            self.freqs = [np.sqrt(len(d)) for d in datasets]
            self.freqs = np.array(self.freqs)
            self.freqs = self.freqs / self.freqs.sum()

        self.cum_lengths = np.cumsum(self.lengths)
        logger.info(f"Total merged dataset length: {self.cum_lengths[-1]}")
        if annealing_schedule is not None:
            self.annealing_schedule = annealing_schedule
            # logger.info(f"Generating indices with annealing schedule: {annealing_schedule}")
            # log each dataset name and ratio start.end
            for i, d in enumerate(datasets):
                logger.info(f"Dataset {d.name}: {annealing_schedule[i]}")
            logger.info(f"Generating indices with annealing schedule.")
            self._check_annealing_schedule()
            self._generate_indices()
            logger.info(f"Generating indices with annealing schedule done.")

    def __len__(self):
        return self.cum_lengths[-1]

    def _check_annealing_schedule(self):
        # check sum-1 property, normalize
        r_start_sum = sum([r_start for r_start, r_end in self.annealing_schedule])
        r_end_sum = sum([r_end for r_start, r_end in self.annealing_schedule])
        
        logger.info(f"Annealing schedule sum: {r_start_sum}, {r_end_sum}, normalizing to 1")
        self.annealing_schedule = [(r_start / r_start_sum, r_end / r_end_sum) for r_start, r_end in self.annealing_schedule]

    def _generate_indices(self):
        # generate deterministic indices
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.indices = []
        self.dataset_indices = []
        # sample one by one
        ratios = [
            torch.linspace(r_start, r_end, steps=len(self)) for r_start, r_end in self.annealing_schedule
        ] # [N, L], N is the number of datasets, L is the length of the merged dataset
        ratios = torch.stack(ratios, dim=1) # [L, N]

        remaining_count = torch.tensor([len(d) for d in self.datasets]) # [N]
        if self.shuffle:
            remaining_indices = [
                torch.randperm(len(d), generator=g).tolist() for d in self.datasets
            ]
        else:
            remaining_indices = [list(range(len(d))) for d in self.datasets]

        for i in range(len(self)):
            # sample dataset 
            current_ratio = ratios[i] * remaining_count 
            current_ratio = current_ratio / current_ratio.sum()
            dataset_idx = torch.multinomial(current_ratio, 1, generator=g).item()
            self.dataset_indices.append(dataset_idx)

            # sample one from the dataset
            remaining_count[dataset_idx] -= 1
            idx = remaining_indices[dataset_idx][remaining_count[dataset_idx]] # take the last one
            self.indices.append(idx)


    def __getitem__(self, idx):
        if hasattr(self, "annealing_schedule"):
            # annealing schedule, to combine without shuffle
            return self.datasets[self.dataset_indices[idx]][self.indices[idx]]
        elif not self.sample_with_sqrt_freq:
            # normal sampling
            for i, cum_len in enumerate(self.cum_lengths):
                if idx < cum_len:
                    return self.datasets[i][idx - (cum_len - self.lengths[i])]
        else:
            # sample with sqrt frequency
            dataset_idx = np.random.choice(len(self.datasets), p=self.freqs)
            return self.datasets[dataset_idx][idx % self.lengths[dataset_idx]]

        raise IndexError("Index out of range")

# ref to https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
class AnnealingDistributedSampler(Sampler):
    def __init__(self, dataset: MergedDataset, annealing_schedule: list[tuple[float, float]], num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.annealing_schedule = annealing_schedule
        self.num_replicas = num_replicas if num_replicas is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = self.generate_indices()
        subset_indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(subset_indices) == self.num_samples
        return iter(subset_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def generate_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for dataset_idx, dataset in enumerate(self.dataset.datasets):
            dataset_indices = list(range(len(dataset)))
            if self.shuffle:
                torch.randperm(len(dataset), generator=g, out=torch.tensor(dataset_indices))

            r_start, r_end = self.annealing_schedule[dataset_idx]
            progress = self.epoch / len(self.dataset)
            current_ratio = r_start + (r_end - r_start) * progress

            num_samples = int(len(dataset) * current_ratio)
            indices.extend([idx + sum(self.dataset.lengths[:dataset_idx]) for idx in dataset_indices[:num_samples]])

        if self.shuffle:
            torch.randperm(len(indices), generator=g, out=torch.tensor(indices))

        # Pad indices to ensure equal distribution across GPUs
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            padding = indices[:padding_size]
            indices += padding

        return indices


def get_word_embedding(model):
    if getattr(model, "fuyu", None) is not None:
        return model.fuyu.language_model.model.embed_tokens
    return model.language_model.model.embed_tokens


def get_output_embedding(model):
    if getattr(model, "fuyu", None) is not None:
        return model.fuyu.language_model.lm_head
    return model.language_model.lm_head


def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:, 0])
    ymin = np.min(obj_pc[:, 1])
    zmin = np.min(obj_pc[:, 2])
    xmax = np.max(obj_pc[:, 0])
    ymax = np.max(obj_pc[:, 1])
    zmax = np.max(obj_pc[:, 2])
    center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    box_size = [xmax - xmin, ymax - ymin, zmax - zmin]
    return center, box_size


def preprocess_sos_eos_for_scan2cap(text: str) -> str:
    if not text.startswith("sos"):
        text = "sos " + text.strip()

    if not text.endswith("eos"):
        text = text.strip() + " eos"

    return text

def postprocess_punctuation_for_caption_metrics(text: str) -> str:
    # add back space before punctuation
    punctuation_chars = [".", ",", "!", "?", ":", ";", "-", "'", "\"", "(", ")", "[", "]", "{", "}", "<", ">", "/", "\\"]
    for p in punctuation_chars:
        text = text.replace(p, f" {p}")

    # remove double spaces
    text = text.replace("  ", " ")
    return text

def clean_answer(data):
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data

if __name__ == "__main__":
    import random

    class DummyDataset(Dataset):
        def __init__(self, size, name):
            self.size = size
            self.name = name

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"{self.name}-{idx}"

    # Create dummy datasets
    datasets = [
        DummyDataset(1000, "A"),
        DummyDataset(2000, "B"),
        DummyDataset(3000, "C"),
    ]

    # Define annealing schedule
    annealing_schedule = [
        (0.6, 0.2),  # Dataset A: start at 60%, end at 20%
        (0.3, 0.3),  # Dataset B: constant 30%
        (0.1, 0.5),  # Dataset C: start at 10%, end at 50%
    ]

    # Create MergedDataset with annealing schedule
    merged_dataset = MergedDataset(
        datasets,
        annealing_schedule=annealing_schedule,
        seed=42,
        shuffle=True
    )

    # Sample and print distribution
    sample_size_step = 500
    distribution = {"A": 0, "B": 0, "C": 0}

    print(f"Sampling {sample_size_step} items at a time from MergedDataset:")
    for i in range(0, len(merged_dataset), sample_size_step):
        end = min(i + sample_size_step, len(merged_dataset))
        for j in range(i, end):
            item = merged_dataset[j]
            dataset_name = item.split("-")[0]
            distribution[dataset_name] += 1

        print(f"\nDistribution after sampling {end} items:")
        for dataset_name, count in distribution.items():
            percentage = (count / end) * 100
            print(f"Dataset {dataset_name}: {count} ({percentage:.2f}%)")
            distribution[dataset_name] = 0

    print("\nFinal distribution:")
    total_samples = len(merged_dataset)
    for dataset_name, count in distribution.items():
        percentage = (count / total_samples) * 100
        print(f"Dataset {dataset_name}: {count} ({percentage:.2f}%)")



    # Verify deterministic behavior
    print("\nVerifying deterministic behavior:")
    another_merged_dataset = MergedDataset(
        datasets,
        annealing_schedule=annealing_schedule,
        seed=42,
        shuffle=True
    )

    all_match = all(
        merged_dataset[i] == another_merged_dataset[i]
        for i in range(len(merged_dataset))
    )
    print(f"All items match: {all_match}")
