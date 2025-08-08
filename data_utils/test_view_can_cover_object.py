

import os
import sys
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
import open_clip
from glob import glob
from tqdm.auto import tqdm
from csv import DictReader
from shapely.geometry import Polygon, MultiPoint
from collections import defaultdict
import json
import multiprocessing
from collections import Counter
import pickle
import uuid
# %%
DSET_VIEW_PATH = '/data/shared/frames_square/'
I2TFILE = 'scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json'
# I2TFILE = 'scene_view_map_resampled_32_8.json'
# TRAIN_ANNO = '/data/mwt/ScanQA-qa/ScanQA_v1.0_val.json'
# TRAIN_ANNO = 'scanrefer/ScanRefer_filtered_train_ScanEnts3D.json'
TRAIN_ANNO = 'scanrefer/ScanEnts3D_Nr3D.csv'


i2t = json.load(open(I2TFILE))

if TRAIN_ANNO.endswith('csv'):
    with open(TRAIN_ANNO) as f:
        annotation_file = list(DictReader(f))
else:
    annotation_file = json.load(open(TRAIN_ANNO))
# annotation_file[0]
# annotation_file = {x['question_id']: x for x in annotation_file}

# with open(TRAIN_ANNO) as f:
#     annotation_file = list(DictReader(f))

print(len(annotation_file), annotation_file[0])

# %%
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

def calculate_duplicate_over_smaller_area_ratio(bbox_pts_2d, h, w):
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

    area_image = image_shape.area
    smaller_area = np.minimum(area, area_image)

    ratio = intersection / smaller_area
    ratio[smaller_area < 1e-6] = 0
    
    return ratio


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

class View:
    def __init__(self, image, pose):
        self.image = image
        self.pose = pose

# %%
class ViewsData:
    def __init__(self, scene_name, view_path='./', bbox_path='./scannet_data/', verbose=False):
        self.verbose = verbose
        self.scene_name = scene_name
        self.view_path = view_path
        # print(scene_name, view_path, bbox_path)
        self.bbox_path = bbox_path

        self.views: Dict[str, 'View'] = {}
        self.view_bbox_overlap = defaultdict(dict) # view_name -> bbox_index -> overlap
        self.bbox_view_overlap = defaultdict(dict) # bbox_index -> view_name -> overlap

        intrinsics = f"{scene_name}/intrinsic_depth.txt"
        self.intrinsics = np.loadtxt(os.path.join(self.view_path, intrinsics))

        self.load_views()
        self.load_bboxes()

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def load_views(self):
        views_color = glob(os.path.join(self.view_path, f'{self.scene_name}/color/*.jpg'))
        self._print(f"Loading {len(views_color)} views from {self.scene_name}")

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
        bbox_path = os.path.join(self.bbox_path, f'{self.scene_name}_bbox.npy')
        
        # bbox = np.load(bbox_path)[..., :6] # xyz, hwl
        bbox = np.load(bbox_path) # xyz, hwl
        self.bbox_object_ids = bbox[..., -1]
        bbox = bbox[..., :6]
        self._print(bbox.shape, bbox[0])  
        if transform_aligned_bbox := True:
            bbox_aligned = bbox.copy()[..., :3]
            bbox_aligned = align(bbox_aligned, axis_align_matrix)
            bbox_aligned = np.concatenate([bbox_aligned, bbox[..., 3:]], axis=-1) # add back hwl
            bbox_corners = calculate_cube_corners(bbox_aligned) # [...,8,3]
            self._print(bbox_corners.shape, bbox_corners[0])
            # unalign
            bbox_corners = align(bbox_corners, np.linalg.inv(axis_align_matrix))
        else:
            bbox_corners = calculate_cube_corners(bbox) # [...,8,3]

        self.bbox = bbox
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
            self._print(f"No legal bbox found in view {view_name}")
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
        # result = calculate_duplicate_area_ratio(bbox_pts_uvd, view.image.size[0], view.image.size[1])
        result = calculate_duplicate_over_smaller_area_ratio(bbox_pts_uvd, view.image.size[0], view.image.size[1])
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

    def run_all_overlap(self):
        for view_name in self.views.keys():
            self.calculate_bbox_overlap(view_name)
    
    def run(self):
        for view_name in self.views.keys():
            self.calculate_bbox_overlap(view_name)
        # print(self.view_bbox_overlap["100"])
        max_bbox_to_view = {}
        # for bbox_index in sorted(self.bbox_view_overlap.keys()):
        for bbox_index in range(self.bbox_corners.shape[0]):
            max_view_name, max_overlap = self.get_max_overlap_view(bbox_index)
            max_bbox_to_view[bbox_index] = max_view_name
            if max_view_name is None:
                self._print(f"bbox {bbox_index} has no matched view")
            else:
                self._print(f"bbox {bbox_index} max overlap: {max_overlap} with view {max_view_name}")

        return max_bbox_to_view
        

    @property
    def view_names(self):
        return list(self.views.keys())


    def from_dict(self, data):
        self.view_bbox_overlap = data['view_bbox_overlap']
        self.bbox_view_overlap = data['bbox_view_overlap']
        self.bbox_object_ids = data['bbox_object_ids']
        self.bbox = data['bbox']
        self.bbox_corners = data['bbox_corners']

        return self
    
    def to_dict(self):
        return {
            'view_bbox_overlap': self.view_bbox_overlap,
            'bbox_view_overlap': self.bbox_view_overlap,
            'bbox_object_ids': self.bbox_object_ids,
            'bbox': self.bbox,
            'bbox_corners': self.bbox_corners,
            'scene_name': self.scene_name,
        }


scene_bbox_view_map = defaultdict(dict) # scene_name -> bbox_index -> view_name
view_path = DSET_VIEW_PATH
# scene_names = glob(f"/scratch/generalvision/ScanQA-feature/frames_square/*")
scene_names = glob(os.path.join(view_path, "*"))
scene_names = [x for x in scene_names if os.path.isdir(x)]
print(f"Found {len(scene_names)} scenes")
scene_view_data = {}

def process_scene(*args):
    scene_name, view_path = args[0]
    scene_name = os.path.basename(scene_name)
    # scene_id, scene_subid = scene_name.split("scene")[-1].split("_")
    # scene_id = int(scene_id)
    # scene_subid = int(scene_subid)

    # print(f"Processing {scene_name}")

    try:
        views_data = ViewsData(scene_name, view_path)
    except FileNotFoundError as e:
        print(e)
        print(f"Scene {scene_name} not found, might be a test scene")
        return None
    views_data.run_all_overlap()
    return scene_name, views_data

with multiprocessing.Pool(processes=64) as pool:
    results = []
    with tqdm(total=len(scene_names), desc="Processing scenes") as pbar:
        for result in pool.imap_unordered(process_scene, [(scene_name, view_path) for scene_name in scene_names]):
            if result:
                scene_name, views_data = result
                scene_view_data[scene_name] = views_data
            pbar.update(1)

# dump to file
with open('scene_view_object_overlap_data.pkl', 'wb') as f:
    pickle.dump({
        scene_name: views_data.to_dict() for scene_name, views_data in scene_view_data.items()
    }, f)

# with multiprocessing.Pool(processes=64) as pool:
#     results = list(tqdm(pool.map(process_scene, [(scene_name, view_path) for scene_name in scene_names]), total=len(scene_names)))

# for result in results:
#     if result:
#         scene_name, views_data = result
#         scene_view_data[scene_name] = views_data


# for scene_name in tqdm(scene_names):
#     scene_name = os.path.basename(scene_name)
#     scene_id, scene_subid = scene_name.split("scene")[-1].split("_")
#     scene_id = int(scene_id)
#     scene_subid = int(scene_subid)

#     print(f"Processing {scene_name}")

#     try:
#         views_data = ViewsData(scene_name, view_path)
#     except FileNotFoundError as e:
#         print(e)
#         print(f"Scene {scene_name} not found, might be a test scene")
#         continue
#     # result = views_data.run()
#     views_data.run_all_overlap()
#     scene_view_data[scene_name] = views_data

# exit(0)
    


# %%
scene_view_counts = []
scene_bbox_counts = []
scene_visible_bbox_percentages = []

for scene_name, views_data in scene_view_data.items():
    view_count = len(views_data.views)
    scene_view_counts.append(view_count)
    
    bbox_count = views_data.bbox.shape[0]
    scene_bbox_counts.append(bbox_count)
    
    # 计算至少从一个视图可见的边界框数量
    visible_bbox_count = sum(1 for bbox_index in range(bbox_count) if views_data.bbox_view_overlap[bbox_index])
    visible_bbox_percentage = (visible_bbox_count / bbox_count) * 100 if bbox_count > 0 else 0
    scene_visible_bbox_percentages.append(visible_bbox_percentage)
    
    print(f"Scene {scene_name}: {view_count} views, {bbox_count} bounding boxes, {visible_bbox_percentage:.2f}% visible")

# 视图数量统计
average_view_count = np.mean(scene_view_counts)
median_view_count = np.median(scene_view_counts)
min_view_count = np.min(scene_view_counts)
max_view_count = np.max(scene_view_counts)

print(f"\nView count statistics:")
print(f"Average view count per scene: {average_view_count:.2f}")
print(f"Median view count per scene: {median_view_count:.2f}")
print(f"Minimum view count: {min_view_count}")
print(f"Maximum view count: {max_view_count}")

# 边界框数量统计
average_bbox_count = np.mean(scene_bbox_counts)
median_bbox_count = np.median(scene_bbox_counts)
min_bbox_count = np.min(scene_bbox_counts)
max_bbox_count = np.max(scene_bbox_counts)
total_bbox_count = np.sum(scene_bbox_counts)

print(f"\nBounding box count statistics:")
print(f"Average bounding box count per scene: {average_bbox_count:.2f}")
print(f"Median bounding box count per scene: {median_bbox_count:.2f}")
print(f"Minimum bounding box count: {min_bbox_count}")
print(f"Maximum bounding box count: {max_bbox_count}")
print(f"Total bounding box count across all scenes: {total_bbox_count}")

# 可见边界框百分比统计
average_visible_bbox_percentage = np.mean(scene_visible_bbox_percentages)
median_visible_bbox_percentage = np.median(scene_visible_bbox_percentages)
min_visible_bbox_percentage = np.min(scene_visible_bbox_percentages)
max_visible_bbox_percentage = np.max(scene_visible_bbox_percentages)

print(f"\nVisible bounding box percentage statistics:")
print(f"Average visible bounding box percentage: {average_visible_bbox_percentage:.2f}%")
print(f"Median visible bounding box percentage: {median_visible_bbox_percentage:.2f}%")
print(f"Minimum visible bounding box percentage: {min_visible_bbox_percentage:.2f}%")
print(f"Maximum visible bounding box percentage: {max_visible_bbox_percentage:.2f}%")

# 视图数量分布
view_count_distribution = Counter(scene_view_counts)
print("\nView count distribution:")
for count, frequency in sorted(view_count_distribution.items()):
    print(f"{count} views: {frequency} scenes")

# 边界框数量分布
bbox_count_distribution = Counter(scene_bbox_counts)
print("\nBounding box count distribution:")
for count, frequency in sorted(bbox_count_distribution.items()):
    print(f"{count} bounding boxes: {frequency} scenes")

# 可见边界框百分比分布
visible_bbox_percentage_bins = [0, 20, 40, 60, 80, 100]
visible_bbox_percentage_distribution = np.histogram(scene_visible_bbox_percentages, bins=visible_bbox_percentage_bins)
print("\nVisible bounding box percentage distribution:")
for i in range(len(visible_bbox_percentage_bins) - 1):
    bin_start = visible_bbox_percentage_bins[i]
    bin_end = visible_bbox_percentage_bins[i+1]
    count = visible_bbox_percentage_distribution[0][i]
    print(f"{bin_start}% - {bin_end}%: {count} scenes")


# %%
def load_scene_images(scene_id):
    images = []
    scene_frame_path = os.path.join(DSET_VIEW_PATH, scene_id, 'color', "*.jpg")
    for image_name in glob(scene_frame_path):
        image = Image.open(image_name)
        image = preprocess(image).cuda()
        images.append(image)
    return torch.stack(images)

def load_single_view(scene_id, view_id):
    view_id = f"{view_id}.jpg" if not view_id.endswith(".jpg") else view_id
    scene_frame_path = os.path.join(DSET_VIEW_PATH, scene_id, 'color', f"{view_id}")
    image = Image.open(scene_frame_path).convert('RGB')
    return image

# %%
import ast
def filter_relative_objects(entities: dict[int, list[str]]):
    # entities = sum(entities.values(), [])
    if isinstance(entities, str):
        # entities = json.loads(entities)
        entities = ast.literal_eval(entities)
    for item in entities:
        
        for entity in item[1]:
            object_id, object_name = entity.split("_", 1)
            object_id = int(object_id)
            # if object_name in ["wall", "floor", "ceiling", "shower_wall"]:
            #     continue
            if any(x in object_name for x in ["wall", "floor", "ceiling"]):
                continue

            yield object_id, object_name

# %%
cdfs = {}
hard_samples = defaultdict(list)
best_views = {}

# %%
# inspect view overlaps
# for annotation in annotation_file:
# I2TFILE = 'scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json'
I2TFILES = {
    # 'original': 'scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json',
    'scanrefer': 'scene_best_view_for_grounding.pth',
    # 'resampled_32_4': 'scene_view_map_resampled_32_4.json',
    # 'resampled_32_6': 'scene_view_map_resampled_32_6.json',
    # 'resampled_32_8': 'scene_view_map_resampled_32_8.json',
    # 'resampled_64_12': 'scene_view_map_resampled_64_12.json',
    # 'resampled_64_8': 'scene_view_map_resampled_64_8.json',
    # 'resampled_64_6': 'scene_view_map_resampled_64_6.json',
    # 'resampled_64_4': 'scene_view_map_resampled_64_4.json',
    # 'resampled_24_8': 'scene_view_map_resampled_24_8.json',
    # 'resampled_24_6': 'scene_view_map_resampled_24_6.json',
    # 'resampled_24_4': 'scene_view_map_resampled_24_4.json',
}
# I2TFILE = 'scene_view_map_resampled_32_4.json'
# I2TFILE = I2TFILES['resampled_32_8']
OVERLAP_THRESHOLD = 0.5
for key, I2TFILE in I2TFILES.items():
    total_bbox = 0
    if key in cdfs:
        continue

    print(f"Processing {I2TFILE}, key: {key}")
    if I2TFILE.endswith('.pth'):
        i2t = torch.load(I2TFILE, map_location='cpu')
    else:
        i2t = json.load(open(I2TFILE))

    if 'view' in i2t:
        i2t = i2t['view']

    TOPK = 500
    
    have_any_effective_acc = np.zeros(TOPK)
    at_least_one_effective_acc = np.zeros(TOPK)
    all_effective_acc = np.zeros(TOPK)
    total = 0
    # for annotation in annotation_file:
    matched = 0
    for annotation in tqdm(annotation_file):
        if 'scene_id' in annotation:
            scene_name = annotation['scene_id']
        else:
            stimulus_id = annotation['stimulus_id']
            scene_name = stimulus_id.split("-")[0]

        question_id = annotation['question_id'] if 'question_id' in annotation else uuid.uuid4().hex
        
        if "object_ids" in annotation:
            bbox_indices = annotation['object_ids']
            question = annotation['question']
            answer = annotation['answers']
            object_names = annotation['object_names']

        else:
            # bbox_indices = [annotation['object_id']]
            result = list(filter_relative_objects(annotation['entities']))
            if len(result) == 0:
                print(f"No object found in {annotation}")
                continue

            bbox_indices, object_names = zip(*result)
            answer = ''

            question = annotation['description'] if 'description' in annotation else annotation['utterance']

        # intify all
        bbox_indices = [int(x) for x in bbox_indices]
        
        # matched_views = i2t['view'][annotation['question_id']] if 'view' in i2t else i2t[annotation['question_id']]

        total_bbox += len(bbox_indices)

        if scene_name not in scene_view_data:
            # print(f"Scene {scene_name} not found")
            continue

        views_data = scene_view_data[scene_name]
        
        # print(f"Scene {scene_name}, question: {question} {answer}")
        # print(f"objects: {bbox_indices}, object names: {object_names}")
        # print(f"top-10 matched view: {matched_views[:10]}")

        view_bbox_overlap = views_data.view_bbox_overlap # view_name -> bbox_index -> overlap
        # print(view_bbox_overlap.keys())
        images, overlaps = [None] * TOPK, [[] for _ in range(TOPK)]
        # for i, kth_view in enumerate(matched_views[:TOPK]):
        for i, kth_view in enumerate(view_bbox_overlap.keys()):
            bbox_overlaps = view_bbox_overlap[kth_view.split('.')[0]] # bbox_index -> overlap
            # print(bbox_overlaps.keys())
            for bbox_index in bbox_indices:
                try:
                    bbox_index_in_data = views_data.bbox_object_ids.tolist().index(bbox_index)
                except ValueError:
                    print(f"bbox {bbox_index} (object {object_names[bbox_indices.index(bbox_index)]}) not found in data")
                if bbox_index_in_data in bbox_overlaps:
                    overlap = bbox_overlaps[bbox_index_in_data]
                    # print(f"bbox {bbox_index} overlap with top-{i} view: {overlap}")
                    # visualize the view
                    # image = load_single_view(scene_name, kth_view)
                    # images.append(image)
                    # overlaps.append(overlap)
                    # images[i] = image
                    # overlaps[i] *= overlap
                    overlaps[i].append(overlap)
                else:
                    overlaps[i] = [0] # have no overlap with this bbox
        
        overlaps_mean = [np.mean(x) if len(x) > 0 else 0 for x in overlaps]
        overlaps_min = [np.min(x) if len(x) > 0 else 0 for x in overlaps]
        overlaps_max = [np.max(x) if len(x) > 0 else 0 for x in overlaps]
        overlaps = overlaps_mean
        best_view = np.argmax(overlaps)
        best_views[question_id] = list(view_bbox_overlap.keys())[best_view]
        if best_view == 0:
            matched += 1

        effective_overlap = np.array(overlaps) > OVERLAP_THRESHOLD
        # consider top-K@threshold overlapped view acc
        have_any_effective = np.cumsum(effective_overlap) > 0 # if any of the top-K views have overlap > threshold

        at_least_one_effective = np.any(np.array(overlaps_max) > OVERLAP_THRESHOLD) # if any of the top-K views have overlap > threshold
        all_effective = np.any(np.array(overlaps_min) > OVERLAP_THRESHOLD) # if all of the top-K views have overlap > threshold

        all_effective_acc += all_effective
        at_least_one_effective_acc += at_least_one_effective

        is_hard = np.sum(effective_overlap) == 0
        if is_hard:
            print(f"Hard sample: {scene_name}, question: {question} {answer}")
            print(f"Max mean overlap: {np.max(overlaps)}")

            hard_samples[key].append(annotation)

        have_any_effective_acc += have_any_effective
        total += 1

        # images, overlaps should be of length TOPK * len(bbox_indices)
        # for object_id in bbox_indices:
        #     bbox_index_in_data = views_data.bbox_object_ids.tolist().index(object_id)
        #     print(bbox_index_in_data)
        #     for i, kth_view in enumerate(matched_views[:20]):
        #         bbox_overlaps = view_bbox_overlap[kth_view.split('.')[0]]
        #         print(bbox_overlaps.keys())
        #         if bbox_index_in_data in bbox_overlaps:
        #             overlap = bbox_overlaps[bbox_index_in_data]
        #             print(f"bbox {object_id} overlap with top-{i} view: {overlap}")
        #             # visualize the view
        #             image = load_single_view(scene_name, kth_view)
        #             images.append(image)
        #             overlaps.append(overlap)
                    # break
        # visualize the images and overlaps in one image
        if visualize := False and len(images) > 0:
            max_cols = 4
            max_rows = (len(images) + max_cols - 1) // max_cols
            print(max_rows, max_cols)
            fig, axs = plt.subplots(max_rows, max_cols, figsize=(max_cols*4, max_rows*4))
            for i, (image, overlap) in enumerate(zip(images, overlaps)):
                ax = axs[i // max_cols, i % max_cols] if max_rows > 1 else axs[i % max_cols]
                ax.imshow(image)
                ax.set_title(f"overlap: {overlap:.2f}")
                ax.axis('off')
            # set figure title
            fig.suptitle(f"{scene_name}, question: {question} {answer} || objects: {bbox_indices}, object names: {object_names}")
            plt.tight_layout()
            plt.show()

        # break
    print(f"Matched: {matched} / {total}")
    # print(have_any_effective_acc / total)
    print(f"Can be mostly seen (mean IoSA > {OVERLAP_THRESHOLD}): {have_any_effective_acc}/{total} = {have_any_effective_acc / total}")
    print(f"At least one object can be seen: {at_least_one_effective_acc}/{total} = {at_least_one_effective_acc / total}")
    print(f"All object can be seen: {all_effective_acc}/{total} = {all_effective_acc / total}")
    # print(total_bbox / total)
    print(f"Total bbox: {total_bbox}, total questions: {total}, average bbox per question: {total_bbox / total}")
    cdfs[key] = have_any_effective_acc / total


# %%
for key in hard_samples.keys():
    print(f"{len(hard_samples[key])} hard samples in {key}, {len(hard_samples[key])/len(annotation_file)*100:.2f}%")

# save
with open(f"ScanQA_val_hard_samples_{OVERLAP_THRESHOLD}.json", "w") as f:
    json.dump(hard_samples["original"], f, indent=4)


# %%
best_views_save = {
    k: [f"{v}.jpg"] for k, v in best_views.items()
}
with open("scene_eval_oracle_qa.json", "w") as f:
    json.dump({"view": best_views_save}, f, indent=4)

# %%
# 1050/len(annotation_file)

# %%
# print max non-1 overlap
np.set_printoptions(threshold=np.inf)
print(cdfs[key])

# %%
(1-0.99910153) * len(annotation_file)

# %%
print(cdfs)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your data in the 'data' variable
# data = have_any_effective_acc / total
data = cdfs['original']

data[data==1.] = 0
data[data==0] = data.max()

# Calculate PDF (differences between two discrete points of CDF)
pdf = np.diff(data, prepend=0)

# Set up the plot style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Create the plot
fig, ax1 = plt.subplots(figsize=(16, 8), dpi=200)

# Plot CDF, shifting x-axis by 1
cdf_line = ax1.plot(range(1, len(data)+1), data, color='blue', linewidth=2, label='Cumulative')
ax1.fill_between(range(1, len(data)+1), data, alpha=0.3, color='blue')
ax1.set_xlabel('K', fontsize=20)
ax1.set_ylabel('Cumulative Solvablity', fontsize=20, color='blue')

# Create a twin axis for PDF
ax2 = ax1.twinx()

# Plot PDF as bars, shifting x-axis by 1
pdf_bars = ax2.bar(range(1, len(pdf)+1), pdf, alpha=0.5, color='red', label='View-wise Additional')
ax2.set_ylabel('View-wise Additional Solvablity', fontsize=20, color='red')

# Align y-axes
y1_min, y1_max = ax1.get_ylim()
y2_min, y2_max = ax2.get_ylim()

# Set the same limits for both axes
combined_min = min(y1_min, y2_min)
combined_max = max(y1_max, y2_max)
combined_min = max(combined_min, 0)
combined_max = max(combined_max, 1)
ax1.set_ylim(combined_min, combined_max)
ax2.set_ylim(combined_min, combined_max)

# Set the same ticks for both axes
ticks = np.linspace(combined_min, combined_max, 11)
ax1.set_yticks(ticks)
ax2.set_yticks(ticks)

# Format tick labels
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

# Customize the plot
plt.title(f'Solvable with top-K view@IoSA>{OVERLAP_THRESHOLD}', fontsize=24, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Set x-axis ticks to start from 1
ax1.set_xticks(range(1, len(data)+1, max(1, len(data)//10)))

# Add horizontal and vertical lines for 70%, 80%, and 90%
percentages = [0.7, 0.8, 0.9]
colors = ['green', 'orange', 'purple']
for percentage, color in zip(percentages, colors):
    k_value = np.argmax(data >= percentage) + 1  # +1 因为我们从K=1开始
    ax1.plot([1, k_value], [percentage, percentage], color=color, linestyle='-', alpha=1, linewidth=2)
    ax1.plot([k_value, k_value], [0, percentage], color=color, linestyle='-', alpha=1, linewidth=2)
    ax1.text(k_value, percentage, f'{percentage*100:.0f}% @ K={k_value}', 
             verticalalignment='bottom', horizontalalignment='left',
             color=color, fontweight='bold', fontsize=20)

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=20)

# 设置网格
ax1.grid(True, linestyle='--', alpha=0.7)
ax2.grid(True, linestyle='--', alpha=0.7)
# ax1.set_axisbelow(True)
# ax2.set_axisbelow(True)

# 设置背景色
ax1.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#ffffff')

# Adjust layout and display
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your data in a list of numpy arrays or lists
# cdf_sequences = [data1, data2, data3, data4]  # Replace with your actual data
# sequence_names = ['Sequence 1', 'Sequence 2', 'Sequence 3', 'Sequence 4']  # Replace with your sequence names
cdf_sequences = list(cdfs.values())
for cdf_seq in cdf_sequences:
    cdf_seq[cdf_seq==1.] = 0
    cdf_seq[cdf_seq==0] = cdf_seq.max()
    
sequence_names = list(cdfs.keys())


# Set up the plot style
# plt.style.use('seaborn')
sns.set_style("whitegrid")
colors = sns.color_palette("husl", len(cdf_sequences))

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each CDF sequence
for i, (data, name) in enumerate(zip(cdf_sequences, sequence_names)):
    ax.plot(range(len(data)), data, color=colors[i], linewidth=2, label=name)
    ax.fill_between(range(len(data)), data, alpha=0.1, color=colors[i])

# Customize the plot
ax.set_title('Comparison of CDF Sequences', fontsize=20, fontweight='bold')
ax.set_xlabel('K', fontsize=14)
ax.set_ylabel('Cumulative Probability', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)

# ax.set_ylim(0, 1)
y_min = cdf_sequences[0].min()
print(y_min)
ax.set_ylim(y_min, 1)
# ax.set_yticks(np.linspace(0, 1, 11))
y_ticks = np.linspace(0, 1, 21)
y_ticks = y_ticks[y_ticks >= y_min]
ax.set_yticks(y_ticks)

# Add a grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Add legend
ax.legend(fontsize=12, loc='lower right')

# Add annotations to highlight differences
# for i in range(1, len(cdf_sequences)):
#     diff = np.array(cdf_sequences[i]) - np.array(cdf_sequences[0])
#     max_diff_index = np.argmax(np.abs(diff))
#     ax.annotate(f'Max diff: {diff[max_diff_index]:.2f}',
#                 xy=(max_diff_index, cdf_sequences[i][max_diff_index]),
#                 xytext=(10, 10), textcoords='offset points',
#                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# Adjust layout and display
plt.tight_layout()
plt.show()


# %%



