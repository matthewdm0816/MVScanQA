import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm.auto import tqdm
from shapely.geometry import Polygon, MultiPoint
from collections import defaultdict
import json
import multiprocessing
import pickle

# %%
DSET_VIEW_PATH = '../SVC/frames_square/'
BBOX_PATH = '../SVC/scannet_data/'

# %%
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
    corners = [np.stack(tensor, axis=-1) for tensor in corners] # => 8 [...,3]
    corners = np.stack(corners, axis=-2) # => [...,8,3]
    return corners

def align(pcd, axis_align_matrix):
    pcd_homogeneous = np.ones((*pcd.shape[:-1], 4))
    pcd_homogeneous[..., :3] = pcd[..., :3]
    pcd_aligned = pcd_homogeneous @ axis_align_matrix.transpose() # ...x4, P' = P R^T
    return pcd_aligned[..., :3] / pcd_aligned[..., 3:]

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

class View:
    def __init__(self, image, pose):
        self.image = image
        self.pose = pose

# %%
class ViewsData:
    def __init__(self, scene_name, view_path='./', bbox_path=BBOX_PATH, verbose=False):
        self.verbose = verbose
        self.scene_name = scene_name
        self.view_path = view_path
        self.bbox_path = bbox_path

        self.views: dict[str, 'View'] = {}
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
            image = image.resize((image.width*2, image.height*2), Image.BICUBIC)
            
            pose = np.loadtxt(os.path.join(self.view_path, f'{self.scene_name}/pose/{view_name}.txt'))
            self.views[view_name] = View(image, pose)

    def get_view(self, view_name):
        return self.views[view_name]
    
    def load_bboxes(self):
        try:
            axis_align_matrix = json.load(open("data/alignments.json", "r"))[self.scene_name]
            axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        except (KeyError, FileNotFoundError):
            axis_align_matrix = np.eye(4) # for test scenes

        self.axis_align_matrix = axis_align_matrix

        bbox_path = os.path.join(self.bbox_path, f'{self.scene_name}_bbox.npy')
        
        bbox = np.load(bbox_path)
        self.bbox_object_ids = bbox[..., -1]
        bbox = bbox[..., :6]
        self._print(bbox.shape, bbox[0])  
        if transform_aligned_bbox := True:
            bbox_aligned = bbox.copy()[..., :3]
            bbox_aligned = align(bbox_aligned, axis_align_matrix)
            bbox_aligned = np.concatenate([bbox_aligned, bbox[..., 3:]], axis=-1)
            bbox_corners = calculate_cube_corners(bbox_aligned)
            self._print(bbox_corners.shape, bbox_corners[0])
            bbox_corners = align(bbox_corners, np.linalg.inv(axis_align_matrix))
        else:
            bbox_corners = calculate_cube_corners(bbox)

        self.bbox = bbox
        self.bbox_corners = bbox_corners

    def calculate_bbox_overlap(self, view_name):
        view = self.views[view_name]
        bbox_pts = self.bbox_corners.reshape((-1, 3))
        bbox_pts_uvd = convert_to_uvd(bbox_pts, self.intrinsics, view.pose)
        bbox_pts_uvd = bbox_pts_uvd.reshape((-1, 8, 3))
        legal_bbox = []
        bbox_indices = []
        for i in range(len(bbox_pts_uvd)):
            result, _ = filter_points(bbox_pts_uvd[i], view.image.size[0], view.image.size[1])
            result2, _ = filter_by_depth(bbox_pts_uvd[i], depth_range=(0, 15))
            if result.shape[0] > 0 and result2.shape[0] > 0:
                legal_bbox.append(bbox_pts_uvd[i])
                bbox_indices.append(i)

        if len(legal_bbox) == 0:
            self._print(f"No legal bbox found in view {view_name}")
            return
        
        bbox_pts_uvd = np.stack(legal_bbox, axis=0)[..., :2]
        result = calculate_duplicate_over_smaller_area_ratio(bbox_pts_uvd, view.image.size[0], view.image.size[1])
        for i, bbox_index in enumerate(bbox_indices):
            self.view_bbox_overlap[view_name][bbox_index] = result[i]
            self.bbox_view_overlap[bbox_index][view_name] = result[i]

    def run_all_overlap(self):
        for view_name in self.views.keys():
            self.calculate_bbox_overlap(view_name)
    
    def to_dict(self):
        return {
            'view_bbox_overlap': self.view_bbox_overlap,
            'bbox_view_overlap': self.bbox_view_overlap,
            'bbox_object_ids': self.bbox_object_ids,
            'bbox': self.bbox,
            'bbox_corners': self.bbox_corners,
            'scene_name': self.scene_name,
        }

def process_scene(*args):
    scene_name, view_path = args[0]
    scene_name = os.path.basename(scene_name)
    try:
        views_data = ViewsData(scene_name, view_path)
    except FileNotFoundError as e:
        print(e)
        print(f"Scene {scene_name} not found, might be a test scene")
        return None
    views_data.run_all_overlap()
    return scene_name, views_data

def main():
    view_path = DSET_VIEW_PATH
    scene_names = glob(os.path.join(view_path, "*"))
    scene_names = [x for x in scene_names if os.path.isdir(x)]
    print(f"Found {len(scene_names)} scenes")
    scene_view_data = {}

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
    
    print("Finished generating scene_view_object_overlap_data.pkl")

if __name__ == '__main__':
    main()