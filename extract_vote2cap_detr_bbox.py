import pickle
import torch
import numpy as np
from collections import defaultdict

def get_3d_box_from_corners(corners):
    """
    Convert 8 corners (8, 3) to 6D format [x, y, z, h, w, l]
    """
    min_corner = np.min(corners, axis=0)
    max_corner = np.max(corners, axis=0)
    
    center = (min_corner + max_corner) / 2
    size = max_corner - min_corner
    
    return np.concatenate([center, size])

def flip_corners(corners):
    corners[..., [0, 2, 1]] = corners[..., [0, 1, 2]]
    corners[..., 2] *= -1
    return corners

# 加载原始的.pkl文件
input_file = "../SVC/pc_features/scannetv2-vote2cap-feature_box_features_281d.pkl"
with open(input_file, "rb") as f:
    data = torch.load(f)

# 创建新的字典来存储结果
result = defaultdict(dict)
total_boxes = 0
total_scenes = 0

SAVE_FULL_BOXES = False

for item in data:
    scene_id = item['scene_id']
    if 'box_corners' in item and 'mask' in item:
        box_corners = item['box_corners'].numpy()
        box_mask = item['mask'].numpy()
        box_corners = flip_corners(box_corners)
        
        bbox_id = 0
        for corners, mask in zip(box_corners, box_mask): 
            if mask or SAVE_FULL_BOXES:
                bbox_6d = get_3d_box_from_corners(corners)
                result[scene_id][bbox_id] = {"bbox": bbox_6d, "is_valid": mask}
                bbox_id += 1

        
        total_boxes += bbox_id
        total_scenes += 1

# 计算平均每个场景的边界框数量
avg_boxes_per_scene = total_boxes / total_scenes if total_scenes > 0 else 0

# 将结果保存为新的.pkl文件
output_file = f"../SVC/i2t/scene_bbox_info_for_valtest_vote2cap_detr{'_full' if SAVE_FULL_BOXES else ''}_latest.pkl"
with open(output_file, "wb") as f:
    pickle.dump(dict(result), f)

print(f"处理完成。结果已保存到 {output_file}")
print(f"平均每个场景的边界框数量: {avg_boxes_per_scene:.2f}")
