import torch
import numpy as np
import json
import os
from icecream import ic
from tqdm.auto import tqdm
import glob

MAX_OBJ_NUM = 100
FEAT_DIM = 1024

all_scene_list = "../SVC/scannet_data/*_aligned_vert.npy"
all_scene_list = sorted([os.path.basename(x) for x in glob.glob(all_scene_list)])
all_scene_list = [x.split('_aligned_vert')[0] for x in all_scene_list]
print(f"Number of scenes: {len(all_scene_list)}")

def get_3d_box_normal(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (length,wide,height)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    
        NOTE: this time, the dimensions are 0-3, 1-4, 2-5
    '''
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])

    R = roty(heading_angle)
    # l,w,h = box_size
    l, h, w = box_size # NOTE: thus, the input can be (x, y, z, size_x, size_y, size_z)
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

mask3d_attributes = torch.load('../chatscene_features/scannet_mask3d_train_attributes.pt', map_location='cpu')
mask3d_attributes_val = torch.load('../chatscene_features/scannet_mask3d_val_attributes.pt', map_location='cpu')
mask3d_attributes_test = torch.load('../chatscene_features/scannet_mask3d_test_attributes.pt', map_location='cpu')
mask3d_feats = torch.load('../chatscene_features/scannet_mask3d_uni3d_feats.pt', map_location='cpu') # scene_id_[object_index] -> 3d feature
mask3d_feats_test = torch.load('../chatscene_features/scannet_mask3d_uni3d_feats_test.pt', map_location='cpu') 

mask3d_attributes.update(mask3d_attributes_val) # scene_id -> {'locs': locs, 'objects': object names}
mask3d_attributes.update(mask3d_attributes_test)
mask3d_feats.update(mask3d_feats_test)

print(f"Number of scenes: {len(mask3d_attributes)}")
print(f"Number of features: {len(mask3d_feats)}")

# find missing scenes
missing_scenes = set(all_scene_list) - set(mask3d_attributes.keys())
print(f"Missing scenes: {missing_scenes}")

mask3d_merged_feats = []
# for scene_id, objects in mask3d_attributes.items():
# for scene_id, objects in tqdm(mask3d_attributes.items()):
for scene_id in tqdm(all_scene_list):
    if scene_id not in mask3d_attributes:
        ic(scene_id)
        mask3d_merged_feats.append({
            'scene_id': scene_id,
            'bbox': torch.zeros(MAX_OBJ_NUM, 6),
            'object_names': [''] * MAX_OBJ_NUM,
            'feature': torch.zeros(MAX_OBJ_NUM, FEAT_DIM+6),
            'mask': torch.zeros(MAX_OBJ_NUM),
            'box_corners': torch.zeros(MAX_OBJ_NUM, 8, 3)
        })
        continue
    
    objects = mask3d_attributes[scene_id]
    # ic(scene_id, len(objects['locs']), len(objects['objects']))
    # ic(len([k for k in mask3d_feats.keys() if scene_id in k]))
    locs = objects['locs']
    object_names = objects['objects']
    object_feats = []
    all_bbox_corners = []

    # continue
    for i, object_name in enumerate(object_names):
        if i >= MAX_OBJ_NUM: # only take the first MAX_OBJ_NUM objects
            break
        feat_id = f"{scene_id}_{i:02d}"
        if feat_id not in mask3d_feats:
            ic(feat_id)
            object_feat = torch.zeros(FEAT_DIM)
        else:
            object_feat = mask3d_feats[f"{scene_id}_{i:02d}"] # 1024
        
        object_loc = locs[i].numpy() # 6
        object_feat = torch.cat([object_feat, torch.tensor(object_loc)], dim=0) # 3d feature + 3d location, 1024+6
        object_feats.append(object_feat)

        bbox_corners = get_3d_box_normal(object_loc[3:], 0, object_loc[:3]) # (8, 3)
        all_bbox_corners.append(bbox_corners) # (n_objects, 8, 3)ikkj

    mask3d_merged_feats.append({
        'scene_id': scene_id,
        'bbox': locs, 
        'object_names': object_names, 
        'feature': torch.stack(object_feats),
        # 'mask': torch.ones(len(object_names)),
        'mask': torch.ones(len(object_feats)),
        'box_corners': torch.tensor(all_bbox_corners)
    })

torch.save(mask3d_merged_feats, '../chatscene_features/scannet_mask3d_trainval_feat+bbox_feats.pt')