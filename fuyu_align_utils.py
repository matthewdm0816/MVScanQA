import torch
import numpy as np
import logging
from shapely.geometry import MultiPoint, Polygon
from torch.utils.data import DataLoader
import open_clip

import os
import glob
from PIL import Image
from tqdm import tqdm
import json
import torch.nn.functional as F
from iou3d import (
    get_minmax_corners,
    get_3d_box,
    from_minmax_to_corners,
    from_minmax_to_xyzhwl,
    box3d_iou,
    box3d_iou_orthogonal,
    batch_box3d_iou_orthogonal,
    batch_get_minmax_corners,
    batch_from_minmax_to_xyzhwl,
)

logger = logging.getLogger(__name__)

def convert_to_uvd(pcd, intr, pose):
    extr = np.linalg.inv(pose)
    # world to camera
    pts = np.ones((pcd.shape[0], 4))
    pts[:,0:3] = pcd[:,0:3]
    pts = pts @ extr.transpose()
    pts = pts[:, :3] / pts[:,3:]
    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1

    d = z * depth_scale
    u = x * fx / z + cx
    v = y * fy / z + cy
    # return u, v, d
    return np.stack([u, v, d], axis=-1)

def convert_to_uvd_batch_torch(pcd, intr, pose):
    # pcd ~ [B, N, 3]
    # intr ~ [B, 4, 4]
    # pose ~ [B, 4, 4]

    extr = torch.linalg.inv(pose)
    # world to camera
    pts = torch.ones((*pcd.shape[:-1], 4)).to(pcd)
    pts[...,0:3] = pcd
    pts = torch.bmm(pts, extr.transpose(1, 2))
    pts = pts[..., :3] / pts[...,3:]
    x, y, z = pts[...,0], pts[...,1], pts[...,2] # [B, N]

    fx = intr[..., 0, 0].unsqueeze(-1) # [B,1]
    fy = intr[..., 1, 1].unsqueeze(-1)
    cx = intr[..., 0, 2].unsqueeze(-1)
    cy = intr[..., 1, 2].unsqueeze(-1)
    depth_scale = 1

    d = z * depth_scale # [B, N]
    u = x * fx / z + cx # [B, N]
    v = y * fy / z + cy
    return torch.stack([u, v, d], dim=-1) # [B, N, 3]

def filter_points(points_uvd, w, h):
    # filter out points outside the image
    mask = (points_uvd[...,0] >= 0) & (points_uvd[...,0] < w) & (points_uvd[...,1] >= 0) & (points_uvd[...,1] < h)
    # filter out points with depth <= 0
    mask = mask & (points_uvd[...,2] > 0)
    return points_uvd[mask], mask

def filter_by_depth(points_uvd, depth_range=(0, 1)):
    mask = (points_uvd[...,2] >= depth_range[0]) & (points_uvd[...,2] < depth_range[1])
    return points_uvd[mask], mask

def calculate_iosa(bbox_pts_2d, h, w):
    # calculates intersection over smaller area

    # bbox_pts_2d: [N, 8, 2]
    # calculate area of each bbox
    # report NAN/INF
    if np.isnan(bbox_pts_2d).any():
        logger.info("Found NAN in bbox_pts_2d")

    if np.isinf(bbox_pts_2d).any():
        logger.info("Found INF in bbox_pts_2d")

    # replace NAN/INF
    bbox_pts_2d = np.nan_to_num(bbox_pts_2d, nan=-100, posinf=-100, neginf=-100)
    bbox_shape = [MultiPoint(bbox).convex_hull for bbox in bbox_pts_2d]
    area = [shape.area for shape in bbox_shape]
    area = np.stack(area, axis=0)
    # print(area.shape)
    # calculate intersection area of each bbox with the image area
    intersection = []
    image_shape = Polygon([[0, 0], [h, 0], [h, w], [0, w]])
    for i in range(len(bbox_pts_2d)):
        intersection.append(bbox_shape[i].intersection(image_shape).area)
    intersection = np.stack(intersection, axis=0)
    min_area = np.minimum(area, h * w)

    # will have 0, if use trim_object and bs >= 2, some box will be padded and have all-same corner therefore area = 0
    # warn if min_area < 1e-6 
    # logger.info(f"min_area: {min_area}, area: {area}, h: {h}, w: {w}")
    # if (min_area < 1e-6).any():
    #     logger.warning(f"min_area: {min_area}, area: {area}, h: {h}, w: {w}")
    #     logger.warning("Found min_area < 1e-6")

    # if min_area 0, set to intersection to 0
    ia = intersection / min_area
    ia[min_area < 1e-6] = 0
    # calculate iosa
    # return intersection / np.minimum(area, h * w)
    ia[np.isnan(ia)] = 0
    return ia

def align(pcd, axis_align_matrix):
    # pcd ~ [B, N, 3]
    # axis_align_matrix ~ [B, 4, 4]
    pcd_homogeneous = torch.ones((*pcd.shape[:-1], 4)).to(pcd) # [B, N, 4]
    pcd_homogeneous[..., :3] = pcd[..., :3]
    # pcd_aligned = pcd_homogeneous @ axis_align_matrix.transpose() # ...x4, P' = P R^T
    pcd_aligned = torch.bmm(pcd_homogeneous, axis_align_matrix.transpose(1, 2))
    return pcd_aligned[..., :3] / pcd_aligned[..., 3:]


@torch.no_grad()
def calculate_in_view_objects(bbox_corners: torch.Tensor, intrinsics: torch.Tensor, poses: torch.Tensor, axis_alignments: torch.Tensor, iosa_threshold: float = 0.25, image_size=(320, 240)):
    """
    calculate in-view object masks
    bbox_corners: [B, N, 8, 3]
    iosa_threshold: float, threshold for intersection over smaller area
    intrinsics: [B, 4, 4]
    poses: [B, 4, 4]
    axis_alignments: [B, 4, 4]
    """
    intrinsics = intrinsics.to(bbox_corners.dtype)
    poses = poses.to(bbox_corners.dtype)
    axis_alignments = axis_alignments.to(bbox_corners.dtype)

    # logger.info("intrinsics shape: %s", intrinsics.shape)
    # logger.info("poses shape: %s", poses.shape)
    # logger.info("axis_alignments shape: %s", axis_alignments.shape)

    # revert to no-align state, since the bbox is aligned, and views are capture unaligned
    bs, num_bboxes, _, _ = bbox_corners.shape
    bbox_corners = bbox_corners.view(bs, -1, 3) # [B, N*8, 3]

    # separate process each item in batch
    bbox_corners_new = []
    # for i in range(bs):
    #     bbox_corners_new.append(align(bbox_corners[i], torch.linalg.inv(axis_alignments[i])))
    bbox_corners = align(bbox_corners, torch.linalg.inv(axis_alignments))

    bbox_pts_uvd = convert_to_uvd_batch_torch(bbox_corners, intrinsics, poses) # [B, N*8, 3]
    bbox_pts_uvd = bbox_pts_uvd.view(bs, num_bboxes, -1, 3) # [B, N, 8, 3]
    # filter box that: all corners are outside the image or depth <= 0
    _, mask1 = filter_points(bbox_pts_uvd, *image_size) # [B, N, 8]
    _, mask2 = filter_by_depth(bbox_pts_uvd, depth_range=(0, 15)) # [B, N, 8]
    mask = mask1 & mask2
    box_valid_mask = mask.sum(dim=-1) > 0 # [B, N]

    iosa_list = []
    for i in range(bs):
        iosa = calculate_iosa(bbox_pts_uvd[i,...,:2].detach().cpu().numpy(), *image_size)
        iosa_list.append(iosa)
    iosa_list = torch.from_numpy(np.stack(iosa_list, axis=0)).to(bbox_pts_uvd.device) # [B, N]
    iosa_mask = iosa_list > iosa_threshold

    bbox_mask = box_valid_mask & iosa_mask

    # logger.info(f"Found {box_valid_mask.sum()}/{bs * num_bboxes} in-view objects")
    # logger.info(f"Found {bbox_mask.sum()}/{bs * num_bboxes} in-view objects with iosa > {iosa_threshold}")

    return bbox_mask, bbox_pts_uvd

def mutual_iou_2(predictions, gts) -> np.ndarray:
    """
    predictions ~ (K1, 8, 3)
    gts ~ (K2, 6)
    returns (K1, K2) matrix
    """
    iou_matrix = np.zeros((len(predictions), len(gts)))
    # print(predictions.shape, gts.shape)
    preds_minmax = batch_get_minmax_corners(predictions)
    preds_xyzhwl = batch_from_minmax_to_xyzhwl(preds_minmax) # [K1, 6]
    # for i, pred in enumerate(predictions):
    for i, pred in enumerate(preds_xyzhwl):
        # for j, gt in enumerate(gts):
        #     # print(i, j, pred.shape, gt.shape)
        #     pred_minmax = get_minmax_corners(pred)
        #     pred_xyzhwl = from_minmax_to_xyzhwl(pred_minmax)
        #     # iou_matrix[i, j], _ = box3d_iou(
        #     #     get_3d_box(pred_xyzhwl[3:], 0, pred_xyzhwl[:3]), get_3d_box(gt[3:], 0, gt[:3])
        #     # )
        #     iou_matrix[i, j] = box3d_iou_orthogonal(pred_xyzhwl[:6], gt[:6])
        # print(i, pred.shape)
        # # pred = pred.unsqueeze(0).expand(len(gts), -1) # [K2, 6]
        # # pred = pred[np.newaxis, ...].expand(len(gts), -1) # [K2, 6]
        # # pred = np.broadcast_to(pred, (len(gts), pred.shape[-1])) # [K2, 6]
        # pred = np.tile(pred, (len(gts), 1)) # [K2, 6]
        iou_matrix[i] = batch_box3d_iou_orthogonal(pred, gts[:, :6]) # [K2]

    return iou_matrix

@torch.no_grad()
def calculate_related_objects(bbox_corners: torch.Tensor, related_object_bboxes: list[torch.Tensor], iou_threshold: float = 0.25):
    """
    calculate related objects
    bbox_corners: [B, N, 8, 3]
    related_object_bboxes: List[torch.Tensor], list of [N, 6], xyzhwl
    iou_threshold: float, threshold for intersection over union
    """
    result = torch.zeros(bbox_corners.shape[:2], dtype=torch.float32, device=bbox_corners.device)

    bbox_corners = bbox_corners.detach().cpu().numpy()
    # bbox_minmax = get_minmax_corners(bbox_corners)
    # bbox_minmax = get_minmax_corners(bbox_corners) 
    # bbox_xyzhwl = from_minmax_to_xyzhwl(bbox_minmax) # [B, N, 6]

    # max_len = max([len(bboxes) for bboxes in related_object_bboxes])
    # print(related_object_bboxes)
    # print(bbox_corners.shape)
    # print([bboxes.shape for bboxes in related_object_bboxes])

    for i, bboxes in enumerate(related_object_bboxes):
        if bboxes is not None and len(bboxes) > 0:
            # print(bboxes.shape,)
            mutual_iou_matrix = mutual_iou_2(bbox_corners[i], bboxes.detach().cpu().numpy())
            max_iou_for_each_bbox = mutual_iou_matrix.max(axis=1)
            # [N, N_related] => [N]
            max_iou_for_each_bbox = max_iou_for_each_bbox * (max_iou_for_each_bbox > iou_threshold)
            result[i] = torch.from_numpy(max_iou_for_each_bbox).to(result)

    # logger.info(f"Total related_object_bboxes: {sum([len(bboxes) for bboxes in related_object_bboxes])}")
    # logger.info(f"Valid box count@{iou_threshold}/total related box: {result[result > 0].shape[0]}/{sum([len(bboxes) for bboxes in related_object_bboxes])}")
    # logger.info(f"Mean valid IoU@{iou_threshold} from detector: {result[result > 0].mean().item()}")

    return result


def to_minmax_bbox(bbox_corners):
    """
    bbox_corners: [N, 8, x>=2]
    """
    min_x = bbox_corners[..., 0].min(dim=-1)[0] # [N]
    min_y = bbox_corners[..., 1].min(dim=-1)[0]
    max_x = bbox_corners[..., 0].max(dim=-1)[0]
    max_y = bbox_corners[..., 1].max(dim=-1)[0]
    return torch.stack([min_x, min_y, max_x, max_y], dim=-1) # [N, 4]


# ---- CLIP Features
class SceneViewsPool:
    def __init__(self, DSET_VIEWS_PATH, SCAN_NAMES, preprocess, init: bool = True, eff_images=None, nocheck_blank: bool = False):
        self.images = dict()
        self.preprocess = preprocess
        self.SCAN_NAMES = SCAN_NAMES
        self.DSET_VIEWS_PATH = DSET_VIEWS_PATH
        self.nocheck_blank = nocheck_blank
        print(f"Loading all scene views from {DSET_VIEWS_PATH}...")
        if init:
            self.init(eff_images=eff_images)

    def init(self, num_workers: int = 32, eff_images=None):
        if eff_images is None:
            print("Loading all scene views...")
        if num_workers < 1:
            # Deprecated
            for filename in tqdm(glob.glob(self.path)):
                image_id = self._getid(filename)
                image = self.preprocess(Image.open(filename))
                self.image_dict[image_id] = image
        else:
            from concurrent.futures import (
                ThreadPoolExecutor,
                wait,
            )

            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = []

            total_files = 0
            for scan_name in tqdm(self.SCAN_NAMES):
                self.images[scan_name] = {}
                p = os.path.join(self.DSET_VIEWS_PATH, scan_name)
                filelist = glob.glob(f"{p}/*.jpg")
                if len(filelist) == 0:
                    filelist = glob.glob(f"{p}/color/*.jpg")
                if len(filelist) == 0:
                    print(f"Warning: no images found in {p}!")

                if eff_images is not None:
                    eff_inames = eff_images[scan_name]
                    filelist = list(filter(lambda fname: os.path.basename(fname) in eff_inames, filelist))
                
                total_files += len(filelist)
            print(f"loading {total_files} scene views...")

            pbar = tqdm(total=total_files, miniters=1_000, mininterval=float("inf"))

            for scan_name in self.SCAN_NAMES:
                p = os.path.join(self.DSET_VIEWS_PATH, scan_name)
                filelist = glob.glob(f"{p}/*.jpg")
                if len(filelist) == 0:
                    filelist = glob.glob(f"{p}/color/*.jpg")
                if len(filelist) == 0:
                    print(f"Warning: no images found in {p}!")

                    
                if eff_images is not None:
                    eff_inames = eff_images[scan_name]
                    filelist = list(filter(lambda fname: os.path.basename(fname) in eff_inames, filelist))
                
                for filename in filelist:
                    future = executor.submit(
                        self._load_single_image_mt, scan_name, filename
                    )
                    future.add_done_callback(lambda future: pbar.update(1))
                    futures.append(future)

            wait(futures)

    def _load_single_image_mt(self, scan_name, filename):
        img_name = os.path.basename(filename)
        img = Image.open(filename).convert("RGB")
        # self.images[scan_name][img_name] = self.preprocess(img)
        frame_id = f"{scan_name}|{img_name}"
        self.images[frame_id] = self.preprocess(img)


def compute_clip_view_features(scene_ids=None, scene_view_map=None, topk=1, alternative_ckpt: str="", model_name="ViT-H-14-378-quickgelu", ckpt_name='dfn5b', return_layer=-1):
    r"""
    Get all scene views + CLIP features
    """
    print(f"Loading top-{topk} images")

    local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # --- Init CLIP Model
    logger.info(f"Loading CLIP Model {model_name}-{ckpt_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_name)
    model.visual.output_tokens = True # for grid feature
    model.eval()
    model = model.to(device)
    grid_size = model.visual.grid_size
    logger.info(f"CLIP grid size: {grid_size}")


    assert scene_view_map is not None, "Provide q-view mapping to load less images"  
    eff_images: dict[str, list] = {}  
    for qid, pred in scene_view_map.items():
        scene_id = f"{qid.strip().split('-')[1]}_00"
        image_names = pred[:topk]
        if scene_id not in eff_images:
            eff_images[scene_id] = []
        eff_images[scene_id].extend(image_names)

    # --- Load views
    if scene_ids is None:
        scene_ids = SCAN_NAMES
    pool = SceneViewsPool(DSET_VIEWS_PATH, scene_ids, preprocess=preprocess, eff_images=eff_images)
    images = pool.images

    # --- Encode CLIP image features
    @torch.no_grad()
    def encode_feature(clip_model: open_clip.CLIP, images):
        feature_shape = None
        logging.info("Beginning Encoding Images...")
        image_feat_dict = {}
        for scan_name, img_dict in tqdm(images.items()):
            dataloader = DataLoader(list(img_dict.items()), batch_size=768)
            image_feat_dict[scan_name] = {}
            with torch.no_grad():
                for batch in dataloader:
                    img_names, images = batch
                    images = images.to(device)
                    image_embeds: torch.Tensor = clip_model.encode_image_gridfeature(images, return_layer=return_layer).to(device) # [B, N, C]
                    G = int(image_embeds.size(1) ** 0.5)
                    image_embeds = image_embeds[:, 1:].view(-1, G, G, image_embeds.size(-1)) # remove [CLS], [B, G, G, C]
                    # record cumsum only, for fast mean pooling
                    image_embeds = image_embeds.cumsum(dim=1)
                    image_embeds = image_embeds.cumsum(dim=2)
                    image_embeds = F.pad(image_embeds, (0, 0, 1, 0, 1, 0), "constant", 0) # [B, 1+G, 1+G, C]
                    feature_shape = image_embeds.shape[1:]
                    # feature_shape = image_embeds.shape
                    for i, img_name in enumerate(img_names):
                        image_feat_dict[scan_name][img_name] = image_embeds[i].cpu()
        return image_feat_dict, feature_shape
   
    image_feat_dict, feature_shape = encode_feature(model, images)
    logging.info("Finished Pre-Computation")
    return image_feat_dict, grid_size
