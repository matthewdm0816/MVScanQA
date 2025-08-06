
import os
import sys
import numpy as np
import torch
from matplotlib import pyplot as plt
import open_clip
from glob import glob
from tqdm.auto import tqdm
import torch
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import umap
import json
from typing import Dict, List, Tuple, Union, Any


model_id = "ViT-H-14-378-quickgelu"
pretrained = 'dfn5b'
DSET_VIEW_PATH = '/data/shared/frames_square/'

# I2TFILE = 'scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json'
I2TFILE = 'scene_eval_scanqa_mv.pth'

# annotation_filenames = {
#     'train': '/data/mwt/ScanQA-qa/ScanQA_v1.0_train.json',
#     'val': '/data/mwt/ScanQA-qa/ScanQA_v1.0_val.json',
#     'test_w_obj': '/data/mwt/ScanQA-qa/ScanQA_v1.0_test_w_obj.json',
#     'test_wo_obj': '/data/mwt/ScanQA-qa/ScanQA_v1.0_test_wo_obj.json',
# }

annotation_filenames = {
    'train': 'qa/ScanQA_mv_train_filtered_cleaned.json',
    'val': 'qa/ScanQA_mv_val_filtered_cleaned.json',
}

def load_scene_images(scene_id):
    images = []
    scene_frame_path = os.path.join(DSET_VIEW_PATH, scene_id, 'color', "*.jpg")
    for image_name in glob(scene_frame_path):
        image = Image.open(image_name)
        image = preprocess(image).cuda()
        images.append(image)
    return torch.stack(images)

def chunked_forward(images, forward_fn, chunk_size=128):
    image_chunks = torch.split(images, chunk_size)
    image_features = []
    # for chunk in tqdm(image_chunks):
    for chunk in image_chunks:
        with torch.no_grad():
            image_features.append(forward_fn(chunk))
    return torch.cat(image_features)


def get_qid_images(i2t, qid, topk=20):
    qid_images = []
    original_images = []
    for frame_name in i2t['view'][qid][:topk]:
        scene_frame_path = os.path.join(DSET_VIEW_PATH, scene_id, 'color', frame_name)
        image = Image.open(scene_frame_path)

        # transform = transforms.ToTensor()
        # rescale to square and to tensor
        transform = transforms.Compose([
            transforms.Resize((378, 378), Image.BICUBIC),
            transforms.ToTensor()
        ])
        original_images.append(transform(image))

        image = preprocess(image).cuda()
        qid_images.append(image)
        
    return torch.stack(qid_images), original_images

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def create_digit_image(digit, image_size, font_size):
    # Create a new blank image
    image = Image.new('RGB', image_size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Load a font
    font = ImageFont.truetype("Arial.ttf", font_size)
    
    # Calculate the size and position of the digit
    text_size = draw.textsize(digit, font=font)
    text_x = (image_size[0] - text_size[0]) // 2
    text_y = (image_size[1] - text_size[1]) // 2
    
    # Draw the digit with white color and black border
    draw.text((text_x-1, text_y-1), digit, font=font, fill='red')
    draw.text((text_x+1, text_y-1), digit, font=font, fill='red')
    draw.text((text_x-1, text_y+1), digit, font=font, fill='red')
    draw.text((text_x+1, text_y+1), digit, font=font, fill='red')
    draw.text((text_x, text_y), digit, font=font, fill='white')
    
    # Convert the PIL image to a Torch tensor
    transform = transforms.ToTensor()
    digit_tensor = transform(image)

    return digit_tensor


# Step 3: Overlay the digit image onto the original image
def overlay_digit_on_image(image_tensor, digit_tensor):
    # Calculate the position to overlay the digit image
    _, h, w = image_tensor.size()
    _, digit_h, digit_w = digit_tensor.size() # c, h, w
    x_offset = (w - digit_w) // 2
    y_offset = (h - digit_h) // 2
    
    # Overlay the digit image
    overlay = image_tensor.clone()
    # overlay[:, y_offset:y_offset+digit_h, x_offset:x_offset+digit_w] += digit_tensor
    # overlay non-black pixels
    non_black = (digit_tensor != 0).any(dim=0).unsqueeze(0).expand_as(digit_tensor)
    overlay[:, y_offset:y_offset+digit_h, x_offset:x_offset+digit_w][non_black] = digit_tensor[non_black]
    return overlay

def chunked_forward(images, forward_fn, chunk_size=128):
    image_chunks = torch.split(images, chunk_size)
    image_features = []
    # for chunk in tqdm(image_chunks):
    for chunk in image_chunks:
        with torch.no_grad():
            image_features.append(forward_fn(chunk))
    return torch.cat(image_features)

def sample_1d_cluster(embeddings, n_clusters):
    # to tensor if numpy
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    # first, cluster embeddings
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr', affinity='precomputed')
    affinity = embeddings @ embeddings.T
    affinity = (affinity + 1) / 2  # scale to [0, 1]
    cluster_labels = clustering.fit_predict(affinity.cpu().numpy())

    # sample first image in each cluster
    sampled_embeddings = []
    sampled_indices = []
    for i in range(n_clusters):
        cluster_embeddings = embeddings[cluster_labels == i]
        if len(cluster_embeddings) > 0:
            sampled_embeddings.append(cluster_embeddings[0])
            # sampled_indices.append(i)
            sampled_indices.append(np.where(cluster_labels == i)[0][0])

    # sort sampled indices
    sampled_indices = sorted(sampled_indices)

    return torch.stack(sampled_embeddings), sampled_indices, cluster_labels

class ViewFeaturePool:
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
        self.feature_pool = {}

    def load_scene_images(self, scene_id):
        images = []
        image_names = []
        scene_frame_path = os.path.join(DSET_VIEW_PATH, scene_id, 'color', "*.jpg")
        for image_name in glob(scene_frame_path):
            image = Image.open(image_name)
            image = preprocess(image).cuda()
            images.append(image)
            image_names.append(os.path.basename(image_name))
        return torch.stack(images), image_names

    def get_features(self, scene_id) -> Dict[str, torch.Tensor]:
        if scene_id in self.feature_pool:
            return self.feature_pool[scene_id]

        # images = load_scene_images(scene_id)
        images, image_names = self.load_scene_images(scene_id)
        image_features = chunked_forward(images, self.model.encode_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        self.feature_pool[scene_id] = {
            name: feature for name, feature in zip(image_names, image_features)
        }

        return self.feature_pool[scene_id] 
    
def eval_mean_pairwise_similarity(embeddings):
    # to tensor
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    # return mean pairwise similarity
    embeddings /= embeddings.norm(dim=-1, keepdim=True)
    sims = embeddings @ embeddings.T
    # remove self-similarity
    sims = sims - torch.eye(len(embeddings), device=sims.device)
    sims = sims.flatten()
    sims = sims[sims > 1e-6]  # remove zero similarity
    return sims.mean().item()

def load_i2tfile(i2tfile):
    if i2tfile.endswith('.json'):
        return json.load(open(i2tfile))
    else:
        return torch.load(i2tfile, map_location='cpu')

print(f"Loading model {model_id}-{pretrained}")
model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
model = model.cuda()
tokenizer = open_clip.get_tokenizer(model_id)

print("Loading i2t file")
# i2t = torch.load(I2TFILE, map_location='cpu')
i2t = load_i2tfile(I2TFILE)
# i2t = json.load(open(I2TFILE))
print(f"Total i2t questions: {len(i2t['view'])}")

# Load annotations
annotations = []
for split, filename in annotation_filenames.items():
    anno = json.load(open(filename))
    # annotations.update({x['question_id']: x for x in anno})
    annotations.extend(anno)

print(f"Loaded annotations for {len(annotations)} questions")

view_feature_pool = ViewFeaturePool(model, preprocess)

for RESAMPLE_TOPK in [8, 16, 24, 32]:
    for RESAMPLE_CLUSTERS in [2, 4, 8, 16]:
        if RESAMPLE_CLUSTERS >= RESAMPLE_TOPK:
            continue

        scene_view_map_resampled_file = f'scanqa_mv_scene_view_map_resampled_{RESAMPLE_TOPK}_{RESAMPLE_CLUSTERS}.json'
        # scene_view_map_resampled_file = f'scanqa_resampled_i2t/scanqa_scene_view_map_resampled_{RESAMPLE_TOPK}_{RESAMPLE_CLUSTERS}.json'
        # create parent directory if not exists
        os.makedirs(os.path.dirname(scene_view_map_resampled_file), exist_ok=True)
        if os.path.exists(scene_view_map_resampled_file):
            scene_view_map_resampled = json.load(open(scene_view_map_resampled_file))
        else:
            scene_view_map_resampled = {}
            
        original_pairwise_sim = []
        resampled_pairwise_sim = []
        for data in tqdm(annotations):
            qid = data['question_id']
            scene_id = data['scene_id']
            original_i2t = i2t['view'][qid] # ordered by similarity

            if qid in scene_view_map_resampled:
                continue

            scene_features = view_feature_pool.get_features(scene_id) # dict of image_name -> feature
            original_i2t_features = torch.stack([scene_features[name] for name in original_i2t])
            original_i2t_features = original_i2t_features[:RESAMPLE_TOPK].cpu().numpy()
            sampled_embeddings, sampled_indices, cluster_labels = sample_1d_cluster(original_i2t_features, RESAMPLE_CLUSTERS)
            # print(original_i2t_features.shape, sampled_embeddings.shape)

            original_pairwise_sim.append(eval_mean_pairwise_similarity(original_i2t_features[:RESAMPLE_CLUSTERS]))
            resampled_pairwise_sim.append(eval_mean_pairwise_similarity(sampled_embeddings))
            # print(f"Original sim: {eval_mean_pairwise_similarity(original_i2t_features[:RESAMPLE_CLUSTERS])}, Resampled sim: {eval_mean_pairwise_similarity(sampled_embeddings)}")
            # print(f"Original sim: {original_pairwise_sim[-1]}, Resampled sim: {resampled_pairwise_sim[-1]}")

            sampled_images = [original_i2t[i] for i in sampled_indices]
            # print(sampled_indices, sampled_images)
            scene_view_map_resampled[qid] = sampled_images
            # break

            # save to file
            if len(scene_view_map_resampled) % 500 == 0:
                json.dump(scene_view_map_resampled, open(scene_view_map_resampled_file, 'w'))
                print(f"Mean original sim: {np.mean(original_pairwise_sim)}, Mean resampled sim: {np.mean(resampled_pairwise_sim)}")

        json.dump(scene_view_map_resampled, open(scene_view_map_resampled_file, 'w'))
        print(f"Mean original sim: {np.mean(original_pairwise_sim)}, Mean resampled sim: {np.mean(resampled_pairwise_sim)}")


