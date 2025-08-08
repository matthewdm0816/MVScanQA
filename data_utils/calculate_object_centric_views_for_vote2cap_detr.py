import torch

import numpy as np
import json
from glob import glob
import os
from PIL import Image
# from iou3d import *
from typing import Dict, List, Tuple, Union, Optional, Any
from shapely.geometry import Polygon, MultiPoint
from collections import defaultdict
import pretty_errors
from tqdm.auto import tqdm
from scipy import sparse
import pickle

SCAN_FAMILY_BASE = "/scratch2/generalvision/yangdejie/data/scanfamily/"
MASK_BASE = "/scratch/generalvision/ScanQA-feature/save_mask/"

def calculate_duplicate_area_ratio(bbox_pts_2d, h, w):
    # bbox_pts_2d: [..., 4, 2]
    # calculate area of each bbox
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

    union = []
    for i in range(len(bbox_pts_2d)):
        union.append(bbox_shape[i].union(image_shape).area)
    union = np.stack(union, axis=0)
    # print(intersection.shape)
    return intersection / union


def calculate_cube_corners(bbox):
    # Calculate the coordinates of the 8 corners of the cube
    # bbox: [..., 6], last dim = xyzhwl
    x = bbox[...,0]
    y = bbox[...,1]
    z = bbox[...,2]
    h = bbox[...,3]
    w = bbox[...,4]
    l = bbox[...,5]
    x_min = x - h / 2
    x_max = x + h / 2
    y_min = y - w / 2
    y_max = y + w / 2
    z_min = z - l / 2
    z_max = z + l / 2
    corners = [
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ]
    # corners = [torch.stack(tensor, dim=-1) for tensor in corners] # => 8 [...,3]
    # corners = torch.stack(corners, dim=-2) # => [...,8,3]
    corners = [np.stack(tensor, axis=-1) for tensor in corners] # => 8 [...,3]
    corners = np.stack(corners, axis=-2) # => [...,8,3]
    return corners

def align(pcd, axis_align_matrix):
    pcd_homogeneous = np.ones((*pcd.shape[:-1], 4))
    pcd_homogeneous[..., :3] = pcd[..., :3]
    pcd_aligned = pcd_homogeneous @ axis_align_matrix.transpose() # ...x4, P' = P R^T
    return pcd_aligned[..., :3] / pcd_aligned[..., 3:]

def convert_from_uvd(u, v, d, intr, pose):
    extr = np.linalg.inv(pose)
    if d == 0:
        return None, None, None
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]

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

def filter_points(points_uvd, w, h):
    # filter out points outside the image
    mask = (points_uvd[...,0] >= 0) & (points_uvd[...,0] < w) & (points_uvd[...,1] >= 0) & (points_uvd[...,1] < h)
    # filter out points with depth <= 0
    mask = mask & (points_uvd[...,2] > 0)
    return points_uvd[mask], mask

def filter_by_depth(points_uvd, depth_range=(0, 1)):
    mask = (points_uvd[...,2] >= depth_range[0]) & (points_uvd[...,2] < depth_range[1])
    return points_uvd[mask], mask

def convert_pc_to_box(obj_pc):
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size

# to process Mask3D proposals (instance masks)
def load_scannet(scan_ids, pc_type, load_inst_info):
    scans = {}
    # attribute
    # inst_labels, inst_locs, inst_colors, pcds, / pcds_pred, inst_labels_pred
    for scan_id in scan_ids:
        # load inst
        if load_inst_info := False:
            inst_labels = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_name', '%s.json'%scan_id)))
            inst_labels = [self.cat2int[i] for i in inst_labels]
            inst_locs = np.load(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_loc', '%s.npy'%scan_id))
            inst_colors = json.load(open(os.path.join(SCAN_FAMILY_BASE, 'scan_data', 'instance_id_to_gmm_color', '%s.json'%scan_id)))
            inst_colors = [np.concatenate(
                [np.array(x['weights'])[:, None], np.array(x['means'])],
                axis=1
            ).astype(np.float32) for x in inst_colors]
            scans[scan_id] = {
                'inst_labels': inst_labels, # (n_obj, )
                'inst_locs': inst_locs,     # (n_obj, 6) center xyz, whl
                'inst_colors': inst_colors, # (n_obj, 3x4) cluster * (weight, mean rgb)
            }
        else:
            scans[scan_id] = {}
            
        # load pcd data
        pcd_data = torch.load(os.path.join(SCAN_FAMILY_BASE, "scan_data", "pcd_with_global_alignment", '%s.pth'% scan_id))
        points, colors, instance_labels = pcd_data[0], pcd_data[1], pcd_data[-1]
        colors = colors / 127.5 - 1
        pcds = np.concatenate([points, colors], 1)
        # convert to gt object
        if load_inst_info:
            obj_pcds = []
            for i in range(instance_labels.max() + 1):
                mask = instance_labels == i     # time consuming
                obj_pcds.append(pcds[mask])
            scans[scan_id]['pcds'] = obj_pcds                     
            # calculate box for matching
            obj_center = []
            obj_box_size = []
            for i in range(len(obj_pcds)):
                c, b = convert_pc_to_box(obj_pcds[i])
                obj_center.append(c)
                obj_box_size.append(b)
            scans[scan_id]['obj_center'] = obj_center
            scans[scan_id]['obj_box_size'] = obj_box_size
        
        # load mask
        if pc_type == 'pred':
            '''
            obj_mask_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".mask" + ".npy")
            obj_label_path = os.path.join(os.path.join(SCAN_FAMILY_BASE, 'mask'), str(scan_id) + ".label" + ".npy")
            obj_pcds = []
            obj_mask = np.load(obj_mask_path)
            obj_labels = np.load(obj_label_path)
            obj_labels = [self.label_converter.nyu40id_to_id[int(l)] for l in obj_labels]
            '''
            obj_mask_path = os.path.join(MASK_BASE, str(scan_id) + ".mask" + ".npz")
            obj_label_path = os.path.join(MASK_BASE, str(scan_id) + ".label" + ".npy")
            obj_pcds = []
            obj_mask = np.array(sparse.load_npz(obj_mask_path).todense())[:50, :]
            obj_labels = np.load(obj_label_path)[:50]
            for i in range(obj_mask.shape[0]):
                mask = obj_mask[i]
                if pcds[mask == 1, :].shape[0] > 0:
                    obj_pcds.append(pcds[mask == 1, :])
            scans[scan_id]['pcds_pred'] = obj_pcds
            scans[scan_id]['inst_labels_pred'] = obj_labels[:len(obj_pcds)]
            # calculate box for pred
            obj_center_pred = []
            obj_box_size_pred = []
            for i in range(len(obj_pcds)):
                c, b = convert_pc_to_box(obj_pcds[i])
                obj_center_pred.append(c)
                obj_box_size_pred.append(b)
            scans[scan_id]['obj_center_pred'] = obj_center_pred
            scans[scan_id]['obj_box_size_pred'] = obj_box_size_pred
    print("finish loading scannet data")
    return scans

class View:
    def __init__(self, image, pose):
        self.image = image
        self.pose = pose

class ViewsData:
    def __init__(self, scene_name, view_path='./'):
        self.scene_name = scene_name
        self.view_path = view_path
        self.views: Dict[str, 'View'] = {}
        self.view_bbox_overlap = defaultdict(dict) # view_name -> bbox_index -> overlap
        self.bbox_view_overlap = defaultdict(dict) # bbox_index -> view_name -> overlap

        intrinsics = f"{scene_name}/intrinsic_depth.txt"
        self.intrinsics = np.loadtxt(os.path.join(self.view_path, intrinsics))

        self.load_views()
        self.load_bboxes()

    def load_views(self):
        views_color = glob(os.path.join(self.view_path, f'{self.scene_name}/color/*.jpg'))
        print(f"Loading {len(views_color)} views from {self.scene_name}")

        for view_color in views_color:
            view_name = os.path.basename(view_color).split('.')[0]
            image = Image.open(view_color)
            # x2 resolution
            image = image.resize((image.width*2, image.height*2), Image.BICUBIC)
            
            pose = np.loadtxt(os.path.join(self.view_path, f'{self.scene_name}/pose/{view_name}.txt'))
            self.views[view_name] = View(image, pose)

    def get_view(self, view_name):
        return self.views[view_name]
    
    def load_bboxes(self):
        # lines = open(f'{self.scene_name}.txt').readlines() # FIXME: hardcoded
        # axis_align_matrix = None
        # for line in lines:
        #     if 'axisAlignment' in line:
        #         axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

        # assert axis_align_matrix is not None
        try:
            axis_align_matrix = json.load(open("alignments.json", "r"))[self.scene_name]
            axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        except KeyError:
            axis_align_matrix = np.eye(4) # for test scenes

        self.axis_align_matrix = axis_align_matrix

        # bbox_path = f"/scratch/generalvision/mowentao/ScanQA/data/scannet/scannet_data/{self.scene_name}_bbox.npy" # FIXME: hardcoded
        # bbox = np.load(bbox_path)[..., :6] # xyz, hwl
        # data = load_scannet([self.scene_name], 'pred', False)
        # bbox_aligned_center = data[self.scene_name]['obj_center_pred']
        # bbox_aligned_size = data[self.scene_name]['obj_box_size_pred']
        # bbox_aligned = np.concatenate([bbox_aligned_center, bbox_aligned_size], axis=-1)

        data = pickle.load(open("/scratch/generalvision/mowentao/SVC/i2t/scene_bbox_info_for_valtest_vote2cap_detr.pkl", "rb"))
        bbox_aligned = data[self.scene_name]
        bbox_aligned = np.stack([bbox_aligned[i]["bbox"] for i in range(len(bbox_aligned))])

        # print(bbox.shape, bbox[0])  
        print(bbox_aligned.shape, bbox_aligned[0])

        bbox_corners = calculate_cube_corners(bbox_aligned) # [...,8,3]
        # unalign
        bbox_corners = align(bbox_corners, np.linalg.inv(axis_align_matrix))

        # if transform_aligned_bbox := True:
        #     bbox_aligned = bbox.copy()[..., :3]
        #     bbox_aligned = align(bbox_aligned, axis_align_matrix)
        #     bbox_aligned = np.concatenate([bbox_aligned, bbox[..., 3:]], axis=-1) # add back hwl
        #     bbox_corners = calculate_cube_corners(bbox_aligned) # [...,8,3]
        #     print(bbox_corners.shape, bbox_corners[0])
        #     # unalign
        #     bbox_corners = align(bbox_corners, np.linalg.inv(axis_align_matrix))
        # else:
        #     bbox_corners = calculate_cube_corners(bbox) # [...,8,3]

        self.bbox = bbox_aligned
        self.bbox_corners = bbox_corners

    def calculate_bbox_overlap(self, view_name):
        view = self.views[view_name]
        # calculate the overlap between the bbox and the view
        bbox_pts = self.bbox_corners.reshape((-1, 3))
        bbox_pts_uvd = convert_to_uvd(bbox_pts, self.intrinsics, view.pose)
        # bbox_pts_uvd, mask = filter_points(bbox_pts_uvd, target_view.size[0]*2, target_view.size[1]*2)
        # filter box that: all corners are outside the image or depth <= 0
        bbox_pts_uvd = bbox_pts_uvd.reshape((-1, 8, 3))
        legal_bbox = []
        bbox_indices = []
        for i in range(len(bbox_pts_uvd)):
            result, _ = filter_points(bbox_pts_uvd[i], view.image.size[0], view.image.size[1])
            # all corners must depth > 0
            result2, _ = filter_by_depth(bbox_pts_uvd[i], depth_range=(0, 15))
            # if result.shape[0] > 0 and result2.shape[0] == 8:
            if result.shape[0] > 0 and result2.shape[0] > 0:
                # print(f"found {result.shape[0]} corners")
                legal_bbox.append(bbox_pts_uvd[i])
                bbox_indices.append(i)

        if len(legal_bbox) == 0:
            print(f"No legal bbox found in view {view_name}")
            return
        
        bbox_pts_uvd = np.stack(legal_bbox, axis=0)[..., :2] # [..., 8, 2]
        # get 2D bbox by minmax of u and v
        # min_uv = np.min(bbox_pts_uvd, axis=-2) # [..., 2]
        # max_uv = np.max(bbox_pts_uvd, axis=-2) # [..., 2]
        # min_umax_v = np.stack([min_uv[...,0], max_uv[...,1]], axis=-1) # [..., 2]
        # max_umin_v = np.stack([max_uv[...,0], min_uv[...,1]], axis=-1) # [..., 2]
        # bbox_pts_2d = np.stack([min_uv, min_umax_v, max_uv, max_umin_v], axis=-2) # [..., 4, 2]
        # print(bbox_pts_2d.shape)

        # calculate area of each bbox
        result = calculate_duplicate_area_ratio(bbox_pts_uvd, view.image.size[0], view.image.size[1])
        for i, bbox_index in enumerate(bbox_indices):
            self.view_bbox_overlap[view_name][bbox_index] = result[i]
            self.bbox_view_overlap[bbox_index][view_name] = result[i]

    def get_max_overlap_view(self, bbox_index):
        max_overlap = 0
        max_view_name = None
        for view_name, overlap in self.bbox_view_overlap[bbox_index].items():
            if overlap > max_overlap:
                max_overlap = overlap
                max_view_name = view_name
        return max_view_name, max_overlap
    
    def run(self):
        for view_name in self.views.keys():
            self.calculate_bbox_overlap(view_name)
        # print(self.view_bbox_overlap["100"])
        max_bbox_to_view = {}
        bbox_index_view_data = {}
        # for bbox_index in sorted(self.bbox_view_overlap.keys()):
        for bbox_index in range(self.bbox_corners.shape[0]):
            max_view_name, max_overlap = self.get_max_overlap_view(bbox_index)
            max_bbox_to_view[bbox_index] = max_view_name
            bbox_index_view_data[bbox_index] = {
                "max_view_name": max_view_name,
                "bbox": self.bbox[bbox_index],
            }
            if max_view_name is None:
                print(f"bbox {bbox_index} has no matched view")
            else:
                print(f"bbox {bbox_index} max overlap: {max_overlap} with view {max_view_name}")
                

        return max_bbox_to_view, bbox_index_view_data
        

    @property
    def view_names(self):
        return list(self.views.keys())
    
if __name__ == '__main__':
    scene_bbox_view_map = defaultdict(dict) # scene_name -> bbox_index -> view_name
    scene_bbox_info = defaultdict(dict) # scene_name -> bbox_index -> bbox_info
    view_path = '/scratch/generalvision/ScanQA-feature/frames_square/'
    scene_names = glob(f"/scratch/generalvision/ScanQA-feature/frames_square/*")
    scene_names = [x for x in scene_names if os.path.isdir(x)]
    print(f"Found {len(scene_names)} scenes")
    for scene_name in tqdm(scene_names):
        scene_name = os.path.basename(scene_name)
        scene_id, scene_subid = scene_name.split("scene")[-1].split("_")
        scene_id = int(scene_id)
        scene_subid = int(scene_subid)
        # if scene_id >= 707 or scene_subid != 0:
        #     continue
        print(f"Processing {scene_name}")
        try:
            views_data = ViewsData(scene_name, view_path)
        except FileNotFoundError as e:
            print(e)
            print(f"Scene {scene_name} not found, might be a test scene")
            continue
        result, bbox_info = views_data.run()
        print(result)
        scene_bbox_view_map[scene_name] = result
        scene_bbox_info[scene_name] = bbox_info

    json.dump(scene_bbox_view_map, open("scene_bbox_view_map_for_valtest_vote2cap_detr.json", "w"))
    # json.dump(scene_bbox_info, open("scene_bbox_info_for_valtest_mask3d.json", "w"))
    pickle.dump(scene_bbox_info, open("scene_bbox_info_for_valtest_vote2cap_detr.pkl", "wb"))
    