import transformers
from transformers import FuyuProcessor, FuyuForCausalLM, AutoModelForCausalLM, FuyuConfig
from PIL import Image
import os
import torch
import json
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from argparse import ArgumentParser
import wandb
import numpy as np
from datetime import datetime
import logging
import colorama
import random
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from copy import deepcopy
# from omegaconf import OmegaConf
from utils.pc_utils import random_sampling, rotx, roty, rotz
import pickle
import MinkowskiEngine as ME
from icecream import ic
from models.fuyu_3d import Fuyu3DCausalLM, Fuyu3DCausalLMv2
from collections import OrderedDict
from fuyu_utils import get_optimizer_param_groups_by_names_dict, ScanQASQA3DDataset, random_sampling, rotx, roty, rotz

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

wandb_project = "Kuri3D"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

base_model_name = "fuyu-8b"
project = "scanqa"
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = base_model_name + "-" + project + "-" + datetime_str
output_name = f"{run_name}-{datetime_str}"
output_dir = "/scratch/generalvision/mowentao/kuri3d-output/" + output_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = (
    "[%(asctime)s %(name)s %(levelname)s] "
    "%(message)s"
)
# logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

def parse_args():
    parser = ArgumentParser()
    # Data
    parser.add_argument("--prompt", default="Answer the following VQAv2 question based on the image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    parser.add_argument("--i2t", type=str, default="/scratch/mowentao/BLIP/scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json")
    parser.add_argument("--frame_path", type=str, default="/scratch/generalvision/ScanQA-feature/frames_square/")
    parser.add_argument("--dataset", type=str, default="scanqa")
    parser.add_argument("--sqa_prompt", type=str, default="Answer the following SQA3D question based on the situation and image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    parser.add_argument("--use_augment", action="store_true")
    
    # Optimization
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_3d", type=float, default=5e-5)
    parser.add_argument("--lr_adapter", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--best_criteria", type=str, default="em")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)

    # Logging
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--print_log_step", type=int, default=50)


    # 3D options
    parser.add_argument("--use_3d", action="store_true")
    parser.add_argument("--spatial_patch_size", type=int, default=24)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--pooling_method", type=str, default="max")

    # LORA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")

    # External config
    parser.add_argument("--prompt_config", type=str, default="")

    # Generation/Decode options
    parser.add_argument("--generation_method", type=str, default="greedy")
    # beam search
    parser.add_argument("--num_beams", type=int, default=5)
    # nucleus sampling
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    return parser.parse_args()

def get_model(args):
    # load model. NOTE: in bfloat16
    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    # model_config = FuyuConfig.from_pretrained(model_id)
    # TODO: add 3D options
    # on CPU, will be moved to GPU later by accelerator
    if args.use_3d:
        # model = Fuyu3DCausalLM.from_pretrained(
        #     model_id, 
        #     torch_dtype=torch.bfloat16,
        #     mnet_path="/scratch/generalvision/mowentao/ScanQA/weights.pth",
        #     freeze_mnet=False,
        #     spatial_patch_size=args.spatial_patch_size,
        #     pooling_method=args.pooling_method,
        # )
        model = Fuyu3DCausalLMv2(
            pretrained_args={
                "pretrained_model_name_or_path": model_id,
                "torch_dtype": torch.bfloat16,
            },
            mnet_path="/scratch/generalvision/mowentao/ScanQA/weights.pth",
            freeze_mnet=args.lr_3d <= 1e-8,
            spatial_patch_size=args.spatial_patch_size,
            pooling_method=args.pooling_method,
        )
        model.load_mnet()
    else:
        model = FuyuForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
        )
    return model, processor 

# TODO: load 3d model

def get_peft_fuyu(model, args):
    if args.lora_rank == 0:
        # freeze all parameters
        logger.info(f"No LoRA applied as lora_rank == {args.lora_rank}.")
        logger.info("Freezing all parameters...")
        for p in model.fuyu.parameters():
            p.requires_grad = False
        return model
    
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_mode=False, 
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
        # for persimmon
        # target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] # "dense", "dense_h_to_4h", "dense_4h_to_h"
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=["mnet", "linear_3d"],
    )

    # apply LORA to the LLM
    if getattr(model, "fuyu", None) is not None:
        print(f"Applying LoRA in {type(model)} submodel Fuyu...")
        model.fuyu = get_peft_model(model.fuyu, peft_config)
        print("Trainable parameters in LVLM:")
        model.fuyu.print_trainable_parameters()
    else:
        print(f"Applying LoRA in {type(model)}...")
        model = get_peft_model(model, peft_config) # FIXME: will this apply LoRA to 3D encoder?
        model.print_trainable_parameters()
    return model

def batch_generate(model: FuyuForCausalLM, processor, questions, images, prompt="Answer the following VQAv2 question based on the image:{}", max_new_tokens=80, return_text=True, skip_special_tokens=False):
    # model_inputs = processor(**inputs).to('cuda')
    text = [prompt.format(q) for q in questions]
    model_inputs = processor(text=text, images=images).to('cuda')
    generated = model.generate( **model_inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)[:, -max_new_tokens:]
    
    model_outputs = processor.batch_decode(generated, skip_special_tokens=skip_special_tokens)
    # print(model_outputs)
    prediction = [m.split('\x04 ', 1)[1] if '\x04' in m else '' for m in model_outputs]
    if return_text:
        return prediction
    else:
        return {
            "model_outputs": generated,
            "prediction": prediction
        }

def batch_generate_v2(model: FuyuForCausalLM, model_inputs, max_new_tokens=80, return_text=True, skip_special_tokens=False, generation_config={}):
    # model_inputs = model_inputs.to('cuda')

    generated = model.generate( **model_inputs, **generation_config, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)[:, -max_new_tokens:]

    model_outputs = processor.batch_decode(generated, skip_special_tokens=skip_special_tokens)
    # print(model_outputs)
    prediction = [m.split('\x04', 1)[1].strip() if '\x04' in m else m for m in model_outputs]
    if return_text:
        return prediction
    else:
        return {
            "model_outputs": generated,
            "prediction": prediction
        }

    
def batch_forward(model, processor, questions, answers, images, prompt="Answer the following VQAv2 question based on the image:{}\x04 {}", max_new_tokens=80, return_text=True):
    # model_inputs = processor(**inputs).to('cuda')
    text = [prompt.format(q, a) for q, a in zip(questions, answers)]
    print(text)
    model_inputs = processor(text=text, images=images).to('cuda')
    generated = model(**model_inputs)
    
    return generated


def get_image_for_question(frame_path, scene_to_image, whole_qid):
    scene_id = whole_qid.split("-")[1] # train-xxxx-xxxx
    image = scene_to_image[whole_qid][0] # xxxx.jpg
    image_path = os.path.join(frame_path, f"{scene_id}_00/color/{image}")
    return Image.open(image_path)

class ScanQAImageOnlyDataset(Dataset):
    @classmethod
    def get_annotation_file(cls, split="train"):
        ANNOTAIONS = {
            "test_w_obj": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_test_w_obj.json",
            "test_wo_obj": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_test_wo_obj.json",
            "train": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_train.json",
            "val": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_val.json"
        }
        return ANNOTAIONS[split]

    def __init__(self, frame_path, split="train", ratio=1.0):
        self.frame_path = frame_path
        self.split = split
        self.annotation = json.load(open(self.get_annotation_file(split)))
        if ratio < 1.0:
            self.annotation = self.annotation[:int(len(self.annotation) * ratio)]
        self.i2t = json.load(open(args.i2t))["view"]

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        question = self.annotation[idx]["question"]
        question_id = self.annotation[idx]["question_id"]
        scene_id = self.annotation[idx]["scene_id"]
        answers = self.annotation[idx]["answers"]
        image = get_image_for_question(self.frame_path, self.i2t, self.annotation[idx]["question_id"])

        return {
            "question": question,
            "question_id": question_id,
            "scene_id": scene_id,
            "answers": answers,
            "image": image,
        }

class ScanQADataset(Dataset):
    @staticmethod
    def get_annotation_file(split="train") -> str:
        ANNOTAIONS = {
            "test_w_obj": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_test_w_obj.json",
            "test_wo_obj": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_test_wo_obj.json",
            "train": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_train.json",
            "val": "/scratch/generalvision/mowentao/ScanQA/data/qa/ScanQA_v1.0_val.json"
        }
        return ANNOTAIONS[split]

    @staticmethod
    def get_scene_path() -> str:
        return "/scratch/generalvision/mowentao/ScanQA/data/scannet/scannet_data"

    def get_multiview_path() -> str:
        return "/scratch/generalvision/mowentao/ScanQA/data/scannet/scannet_data/enet_feats_maxpool"

    def __init__(self, frame_path, split="train", ratio=1.0,
                 use_color=True, use_height=False, use_normal=False, use_multiview=False, use_augment=False,
                 quantization_size=0.02, num_points=50_000
                 ):
        self.frame_path = frame_path
        self.split = split
        self.annotation = json.load(open(self.get_annotation_file(split)))
        if ratio < 1.0:
            self.annotation = self.annotation[:int(len(self.annotation) * ratio)]
        self.i2t = json.load(open(args.i2t))["view"]

        self.use_color = use_color
        self.use_height = use_height
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_augment = use_augment

        self.quantization_size = quantization_size # 0.02 ~ MinkowskiEngine default for ScanNet segmentation
        self.num_points = num_points # 50_000 default for ScanRefer
        
        self._load()

    def _load(self): 
        """
        Load 3D scene, instance and object information
        """
        logger.info("Loading ScanQA data...")
        # add scannet data
        # self.scene_list = sorted(list(set([data['scene_id'] for data in self.scanqa])))
        self.scene_list = sorted(list(set([data['scene_id'] for data in self.annotation])))

        # load scene data
        self.scene_data = {}
        scene_path = self.get_scene_path()
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]['mesh_vertices'] = np.load(os.path.join(scene_path, scene_id)+'_aligned_vert.npy') # axis-aligned
            self.scene_data[scene_id]['instance_labels'] = np.load(os.path.join(scene_path, scene_id)+'_ins_label.npy')
            self.scene_data[scene_id]['semantic_labels'] = np.load(os.path.join(scene_path, scene_id)+'_sem_label.npy')
            self.scene_data[scene_id]['instance_bboxes'] = np.load(os.path.join(scene_path, scene_id)+'_aligned_bbox.npy')

    def _augment_pc(self, point_cloud):
        """
        Augment partial point cloud and bounding boxes
        """
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:,0] = -1 * point_cloud[:,0]
            flip_x = -1
        else:
            flip_x = 1
            
        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:,1] = -1 * point_cloud[:,1]
            flip_y = -1
        else:
            flip_y = 1

        # Rotation along X-axis
        rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
        rot_mat = rotx(rot_angle)
        point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

        rot_mat_total = rot_mat[:] # Rx

        # Rotation along Y-axis
        rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
        rot_mat = roty(rot_angle)
        point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

        rot_mat_total = np.dot(rot_mat, rot_mat_total) # RyRx

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
        rot_mat = rotz(rot_angle)
        point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))

        rot_mat_total = np.dot(rot_mat, rot_mat_total) # RzRyRx

        # Translation
        point_cloud, factor = self._translate(point_cloud)

        return point_cloud, factor, rot_mat_total, flip_x, flip_y

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

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        # get question, image, answers
        question = self.annotation[idx]["question"]
        question_id = self.annotation[idx]["question_id"]
        scene_id = self.annotation[idx]["scene_id"]
        answers = self.annotation[idx]["answers"]
        image = get_image_for_question(self.frame_path, self.i2t, self.annotation[idx]["question_id"])

        # get scene data -- adapted from ScanQA
        mesh_vertices = self.scene_data[scene_id]['mesh_vertices'].copy() # supposedly (50000, 9), xyz, rgb, normal
        instance_labels = self.scene_data[scene_id]['instance_labels'].copy()
        semantic_labels = self.scene_data[scene_id]['semantic_labels'].copy()
        instance_bboxes = self.scene_data[scene_id]['instance_bboxes'].copy()
        
        # original_point_cloud = mesh_vertices[:,0:3]

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3]
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            # point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            point_cloud[:,3:6] = point_cloud[:,3:6]/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1) # p (50000, 7)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1)

        # load multiview database
        if self.use_multiview:
            enet_feats_file = os.path.join(self.get_multiview_path(), scene_id) + '.pkl'
            multiview = pickle.load(open(enet_feats_file, 'rb'))
            point_cloud = np.concatenate([point_cloud, multiview],1) # p (50000, 135)

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        if self.use_augment and self.split == "train":
            point_cloud, factor, rot_mat, flip_x, flip_y = self._augment_pc(point_cloud)
        else:
            factor = [0, 0, 0]
            rot_mat = np.eye(3)
            flip_x = 1
            flip_y = 1

        # ic(point_cloud.shape, instance_labels.shape, semantic_labels.shape)
        # ic(point_cloud[:, 3:].max(), point_cloud[:, 3:].min())

        # Minkowski Engine
        coords, feats, labels = ME.utils.sparse_quantize(
            coordinates=point_cloud[:, :3],
            # features=self.normalize_color(point_cloud[:, 3:], is_color_in_range_0_255=True), # FIXME: is this correct? 
            features=self.normalize_color(point_cloud[:, 3:], is_color_in_range_0_255=True), # FIXME: is this correct?
            labels=semantic_labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)

        return {
            # 2D I-Q-A
            "question": question,
            "question_id": question_id,
            "scene_id": scene_id,
            "answers": answers,
            "image": image,
            # 3D
            "coords": coords,
            "feats": feats,
            "labels": labels,
        }

    @staticmethod
    def normalize_color(color: np.ndarray, is_color_in_range_0_255: bool = False) -> np.ndarray:
        """
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

def get_trainval_datasets(args) -> dict[str, Dataset]:
    if args.dataset == "scanqa_sqa":
        assert args.use_3d, "scanqa_sqa only works with 3D"
        train_dataset = ScanQASQA3DDataset(args.frame_path, split="train", ratio=args.train_ratio, quantization_size=args.voxel_size, use_augment=args.use_augment)
        val_dataset = ScanQASQA3DDataset(args.frame_path, split="val", quantization_size=args.voxel_size, use_augment=args.use_augment)
    elif args.use_3d:
        train_dataset = ScanQADataset(args.frame_path, split="train", ratio=args.train_ratio, quantization_size=args.voxel_size, use_augment=args.use_augment)
        val_dataset = ScanQADataset(args.frame_path, split="val", quantization_size=args.voxel_size, use_augment=args.use_augment)
    else:
        train_dataset = ScanQAImageOnlyDataset(args.frame_path, split="train", ratio=args.train_ratio)
        val_dataset = ScanQAImageOnlyDataset(args.frame_path, split="val")
    return {
        "train": train_dataset,
        "val": val_dataset,
    }

def collate_fn(examples) -> dict:
    global processor, PROMPT, SQA_PROMPT
    # print(examples[0])
    # texts = [PROMPT.format(e["question"], random.choice(e["answers"])) for e in examples]
    # if isinstance(PROMPT, list):
    #     texts = [random.choice(PROMPT).format(e["question"], random.choice(e["answers"])) for e in examples]
    # else:
    #     texts = [PROMPT.format(e["question"], random.choice(e["answers"])) for e in examples]
    # print(texts)
    texts = []
    for e in examples:
        if e.get("split", None) == "sqa":
            texts.append(SQA_PROMPT.format(e["situation"], e["question"], random.choice(e["answers"])))
        elif isinstance(PROMPT, list):
            texts.append(random.choice(PROMPT).format(e["question"], random.choice(e["answers"])))
        else:
            texts.append(PROMPT.format(e["question"], random.choice(e["answers"])))
    

    images = [e["image"] for e in examples]
    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True
    )

    positions = []

    x04_token = processor.tokenizer("\x04", add_special_tokens=False)["input_ids"][-1] # first is <s>
    for input_ids in output["input_ids"]:
        # position = (input_ids == processor.tokenizer.vocab["<s>"]).nonzero(as_tuple=True)[0][0]
        position = (input_ids == x04_token).nonzero(as_tuple=True)[0][0]

        positions.append(position - 1) # -1 to consider that 
    # positions = torch.tensor(positions)
    output["labels"] = torch.full_like(output["input_ids"], -100)  # This creates a tensor filled with -100
    for i in range(len(positions)):
        output["labels"][i, positions[i]:] = output["input_ids"][i, positions[i]:] # mask the tokens before <s> in labels
    # print(output["labels"])

    # 3D data
    if "coords" in examples[0]:
        data_3d = [(e["coords"], e["feats"], e["labels"]) for e in examples]
        data_3d = ME.utils.batch_sparse_collate(data_3d)

        output["mnet_inputs"] = data_3d

    return output

def collate_fn_eval(examples) -> dict:
    # for inference, only format question
    global processor, PROMPT, SQA_PROMPT
    if isinstance(PROMPT, list):
        prompt_for_question = random.choice(PROMPT).split("\x04")[0]
    else:
        prompt_for_question = PROMPT.split("\x04")[0] # + "\x04 "

    prompt_for_question_sqa = SQA_PROMPT.split("\x04")[0]
    # texts = [prompt_for_question.format(e["question"]) for e in examples]
    texts = []
    for e in examples:
        if e.get("split", None) == "sqa":
            texts.append(prompt_for_question_sqa.format(e["situation"], e["question"])) # situation already merged in question
        else:
            texts.append(prompt_for_question.format(e["question"]))
            
    # print(texts)
    images = [e["image"] for e in examples]
    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True
    )
    output["answers"] = [e["answers"] for e in examples]

    # 3D data
    if "coords" in examples[0]:
        data_3d = [(e["coords"], e["feats"], e["labels"]) for e in examples]
        data_3d = ME.utils.batch_sparse_collate(data_3d)

        output["mnet_inputs"] = data_3d
    
    return output


def get_trainval_dataloaders(args) -> dict[str, DataLoader]:
    # train_dataloader = torch.utils.data.DataLoader(args.datasets["train"], batch_size=args.batch_size, shuffle=True)
    # val_dataloader = torch.utils.data.DataLoader(args.datasets["val"], batch_size=args.batch_size, shuffle=False)
    dataloaders = {
        split: DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=(split=="train"), 
            collate_fn=collate_fn if split=="train" else collate_fn_eval,
            num_workers=args.num_workers, 
        ) for split, dataset in args.datasets.items()
    }
    return dataloaders

def get_optimizer_scheduler(args, model):
    # params_3d = ["mnet", "linear_3d"]
    # no_decay = ["bias", "layer_norm.weight"]
    name_to_params_group_map = OrderedDict()
    name_to_params_group_map["params_adapter"] = ["linear_3d"]
    name_to_params_group_map["params_3d"] = ["mnet"]
    name_to_params_group_map["no_decay"] = ["bias", "layer_norm.weight"]
    # other params including the LVLM (fuyu)
    lr_dict = {
        "params_3d": args.lr_3d,
        "params_adapter": args.lr_adapter,
        "no_decay": args.lr,
    }
    weight_decay_dict = {
        "params_3d": args.weight_decay,
        "params_adapter": args.weight_decay,
        "no_decay": 0,
    }
    optimizer_param_groups_dict, optimizer_grouped_parameter_names = get_optimizer_param_groups_by_names_dict(
        model,
        name_to_params_group_map,
        lr_dict,
        weight_decay_dict,
        lr_default=args.lr,
        weight_decay_default=args.weight_decay,
    )
    optimizer_grouped_parameters = list(optimizer_param_groups_dict.values())
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in params_3d)  and p.requires_grad],
    #         "weight_decay": args.weight_decay,
    #         "lr": args.lr,
    #     } if args.lr > 1e-8 else {},
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in params_3d) and p.requires_grad],
    #         "weight_decay": args.weight_decay,
    #         "lr": args.lr_3d,
    #     } if args.lr_3d > 1e-8 else {},
    #     {
    #         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #         "weight_decay": 0.0,
    #         "lr": args.lr,
    #     } if args.lr > 1e-8 else {},
    # ]
    # optimizer_grouped_parameters = [value for value in optimizer_grouped_parameters if len(value) > 0]
    # optimizer_grouped_parameter_names = [
    #     {
    #         "params": [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in params_3d)  and p.requires_grad],
    #     } if args.lr > 1e-8 else {},
    #     {
    #         "params": [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in params_3d) and p.requires_grad],
    #     } if args.lr_3d > 1e-8 else {},
    #     {
    #         "params": [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    #     } if args.lr > 1e-8 else {},
    # ]
    logger.info(optimizer_grouped_parameter_names)
    # params_group = {name: {
    #     "params": [],
    #     "weight_decay": value[1],
    #     "lr": value[2],
    # } for name, value in name_to_params_group_map.items()}
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         for name, value in name_to_params_group_map.items():
    #             if any(nd in n for nd in value[0]):
    #                 params_group[name]["params"].append(p)
    #                 break
    # optimizer_grouped_parameters = [value for name, value in params_group.items()]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)
    elif args.scheduler == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented.")
    # sqrt scheduler with warmup
    # scheduler = transformers.get_inverse_sqrt_schedule(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)
    return optimizer, scheduler
    

def train(args, epoch):
    logger.info(f"[{epoch}]start training...")
    args.model.train()
    torch.cuda.empty_cache()
    accelerator: Accelerator = args.accelerator
    model: FuyuForCausalLM = args.model

    for step, batch in enumerate(args.dataloaders["train"]):

        with accelerator.accumulate(model):
            outputs = model(return_dict=True, **batch) 
            loss = outputs.loss
            # report loss to wandb
            # reduce loss over processes for logging purposes
            with torch.no_grad():
                loss_mean = accelerator.gather(loss.unsqueeze(0)).mean()
                args.history_losses.append(loss_mean.item())
                if accelerator.is_local_main_process:
                    wandb.log({
                        "train/loss": args.history_losses[-1], 
                        "train/global_step": len(args.history_losses),
                        "train/lr": args.scheduler.get_last_lr()[0],
                    })

            accelerator.backward(loss)
            if args.gradient_clipping > 1e-8:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            args.optimizer.step()
            args.scheduler.step()
            args.optimizer.zero_grad()

        if (step + 1) % args.print_log_step == 0:
            if accelerator.is_local_main_process:
                print(f"[{epoch}/{step}-{args.total_steps}]loss: {args.history_losses[-1]}, lr: {args.scheduler.get_last_lr()[0]}")

        if len(args.history_losses) % args.checkpointing_steps == 0:
            # save checkpoint
            # only save PEFT lora
            output_dir_ckpt = os.path.join(output_dir, f"ckpt-{len(args.history_losses)}")
            if not os.path.exists(output_dir_ckpt):
                os.makedirs(output_dir_ckpt, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir_ckpt)

def metrics(preds: str, labels: list[str]):
    em = 0
    if preds in labels:
        em = 1

    refined_em = 0
    for label in labels:
        if ''.join(label.split()) in ''.join(preds.split()):
            refined_em = 1
            break
        if ''.join(preds.split()) in ''.join(label.split()):
            refined_em = 1
            break
    
    return em, refined_em
        
def validate(args, epoch):
    logger.info(f"[{epoch}]start validating...")
    args.model.eval()
    torch.cuda.empty_cache()
    accelerator: Accelerator = args.accelerator

    total, correct_em, correct_refined_em = 0, 0, 0
    for step, batch in enumerate(args.dataloaders["val"]):
        with torch.no_grad():
            answers = deepcopy(batch["answers"])
            del batch["answers"]
            # inference by generate
            unwrapped_model = accelerator.unwrap_model(args.model, keep_fp32_wrapper=True)
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_error() # disable logging
            # print(unwrapped_model.forward, getattr(unwrapped_model.forward, "__wrapped__", None))
            pred_answer = batch_generate_v2(
                unwrapped_model, 
                batch,
                # prompt=args.prompt.split("\x04")[0], 
                max_new_tokens=args.max_new_tokens,
                return_text=True,
                skip_special_tokens=True,
                generation_config=args.generation_config,
            )
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_info() # reset to info
            if args.verbose:
                print(pred_answer)
            for i in range(len(pred_answer)):
                em, refined_em = metrics(pred_answer[i], answers[i])
                correct_em += em
                correct_refined_em += refined_em
                total += 1

    # gather results from all processes
    correct_em = accelerator.gather(torch.tensor(correct_em).cuda().unsqueeze(0)).sum().item()
    correct_refined_em = accelerator.gather(torch.tensor(correct_refined_em).cuda().unsqueeze(0)).sum().item()
    total = accelerator.gather(torch.tensor(total).cuda().unsqueeze(0)).sum().item()

    # record best
    metrics_epoch = {
        "em": correct_em/total,
        "refined_em": correct_refined_em/total,
    }
    if args.best_metrics.get(args.best_criteria, -1) < metrics_epoch[args.best_criteria]:
        args.best_metrics = metrics_epoch
        if accelerator.is_local_main_process:
            print(f"New best {args.best_criteria}: {args.best_metrics[args.best_criteria]}")
            # save checkpoint
            output_dir_ckpt = os.path.join(output_dir, f"best-{args.best_criteria}")
            if not os.path.exists(output_dir_ckpt):
                os.makedirs(output_dir_ckpt, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(args.model)
            unwrapped_model.save_pretrained(output_dir_ckpt)
            
    if accelerator.is_local_main_process:
        print(f"[{epoch}]em: {metrics_epoch['em']}, refined_em: {metrics_epoch['refined_em']}")
        wandb.log({
            "eval/em": metrics_epoch["em"],
            "eval/refined_em": metrics_epoch["refined_em"], 
            "train/global_step": len(args.history_losses), 
            "eval/epoch": epoch,
            # best
            "best/em": args.best_metrics["em"],
            "best/refined_em": args.best_metrics["refined_em"],
            }
        )


if __name__ == "__main__":
    args = parse_args()
    # PROMPT = args.prompt
    # decode in utf-8
    if args.prompt_config != "":
        prompt_config = json.load(open(args.prompt_config))
        args.prompt_config = prompt_config
        PROMPT = [
            p + prompt_config["postfix"] for p in prompt_config["prompts"]
        ]
        print(repr(PROMPT))
    else:
        PROMPT = args.prompt.encode().decode('unicode_escape')
        print(repr(PROMPT))

    SQA_PROMPT = args.sqa_prompt.encode().decode('unicode_escape')
    device = args.device

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    args.accelerator = accelerator

    if accelerator.is_local_main_process:
        print(args)

    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")
        transformers.utils.logging.set_verbosity_info()
    else:
        logging.basicConfig(
            format=log_format, level=logging.ERROR, datefmt="%I:%M:%S"
        )
        transformers.utils.logging.set_verbosity_error()
        ic.disable()

    if args.seed is not None:
        set_seed(args.seed)

    # make generation config
    if args.generation_method == "greedy":
        args.generation_config = {}
    elif args.generation_method == "beam_search":
        args.generation_config = {
            "num_beams": args.num_beams,
            "do_sample": False,
        }
    elif args.generation_method == "nucleus":
        args.generation_config = {
            "num_beams": 1,
            "do_sample": True,
            "top_p": args.top_p,
            "top_k": args.top_k,
        }

    accelerator.wait_for_everyone()

    datasets = get_trainval_datasets(args)
    args.datasets = datasets
    dataloaders = get_trainval_dataloaders(args)
    args.dataloaders = dataloaders

    
    model, processor = get_model(args)
    model = get_peft_fuyu(model, args)
    args.model = model
    args.processor = processor

    logger.info(f"Total {len(args.datasets['train'])} training samples.")
    logger.info(f"Total {len(args.datasets['val'])} validation samples.")
    # calc total steps
    args.total_steps = args.epochs * len(args.datasets["train"]) // args.batch_size // accelerator.num_processes
    args.warmup_steps = args.total_steps // 10
    logger.info(f"Total {args.total_steps} training steps.")
    logger.info(f"Total {args.warmup_steps} warmup steps.")

    args.history_losses = []
    args.best_metrics = {}

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    args.optimizer = optimizer
    args.scheduler = scheduler

    # args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"], args.scheduler = accelerator.prepare(
    #     args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"], args.scheduler
    # )
    # NOTE: prepare scheduler will make it step for N_GPU times, which is not what we want
    args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"] = accelerator.prepare(
        args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"]
    )

    # convert ME's batchnorm to sync batchnorm
    args.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(args.model)
    # args.model.module.load_mnet()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(args.datasets["train"])), 3):
        logger.info(f"Sample {index} of the training set: {args.datasets['train'][index]}.")

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(args.datasets['train'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")


    if accelerator.is_local_main_process:
        wandb.init(project=wandb_project, name=run_name, config=args, config_exclude_keys=["accelerator", "model", "processor", "optimizer", "scheduler", "datasets", "dataloaders"])


    for epoch in range(args.epochs):
        train(args, epoch)
        validate(args, epoch)

        # save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        unwrapped_model = accelerator.unwrap_model(args.model)
        unwrapped_model.save_pretrained(output_dir)

    if accelerator.is_local_main_process:
        wandb.finish()