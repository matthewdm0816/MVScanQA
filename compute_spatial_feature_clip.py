import torch
from PIL import Image
import open_clip
import json
import math
from tqdm.auto import tqdm
import os
import numpy as np
import logging
from torch.utils.data import DataLoader

from fuyu_align_utils import SceneViewsPool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SCAN_NAMES = list(
    sorted(
            [
                line.rstrip()
                for line in open(
                    "/home/mowentao/data/ScanQA/data/scannet/meta_data/scannetv2.txt"
                )
            ]
        ),
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="data/frames_square")
    parser.add_argument('--model_id', type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument('--pretrained', type=str, default='dfn5b')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--output_path', type=str, default="data/clip_features")

    os.path.mkdirs(parser.output_path, exist_ok=True)

    args = parser.parse_args()
    return args

class _print_once:
    def __init__(self):
        self.first = True

    def __call__(self, *args, **kwargs):
        if self.first:
            print(*args, **kwargs)
            self.first = False

print_once = _print_once()

@torch.no_grad()
def encode_feature(args, clip_model: open_clip.CLIP, images, device=torch.device("cuda")):
    logging.info("Beginning Encoding Images...")
    dataloader = DataLoader(list(images.items()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    for batch in tqdm(dataloader, total=len(dataloader)):
        image_feat_dict = {}
        img_names, images = batch
        images = images.to(device)
        _, image_tokens = clip_model.visual(images)
        G = int((image_tokens.size(1)-1) ** 0.5)
        print_once(f"Image tokens shape: {image_tokens.shape}, G: {G}")
        image_embeds = image_embeds[:, 1:].view(-1, G, G, image_embeds.size(-1)) # remove [CLS], [B, G, G, C]
        
        for i, img_name in enumerate(img_names):
            image_feat_dict[img_name] = image_embeds[i].cpu().numpy()

        yield image_feat_dict

def main(args):
    local_rank = 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # --- Init CLIP Model
    logger.info(f"Loading CLIP Model {args.model_id}-{args.pretrained}...")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_id, pretrained=args.pretrained)
    model.visual.output_tokens = True # for grid feature
    model.eval()
    model = model.to(device)
    grid_size = model.visual.grid_size
    logger.info(f"CLIP grid size: {grid_size}")


    pool = SceneViewsPool(args.image_path, SCAN_NAMES[:10], preprocess=preprocess)
    images = pool.images # [scan name -> frame id -> image]

    for image_feat_dict in encode_feature(args, model, images, device):
        for img_name, img_feat in image_feat_dict.items():
            output_path = os.path.join(args.output_path, img_name + ".npy")
            np.save(output_path, img_feat)


if __name__ == "__main__":
    args = parse_args()
    main(args)