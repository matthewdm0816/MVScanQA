import transformers
from transformers import FuyuProcessor, FuyuForCausalLM, AutoModelForCausalLM, FuyuConfig, AutoTokenizer, BertTokenizer
from PIL import Image
import os
import torch
import json
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.peft_model import PeftModel
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
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import deepspeed

from accelerate.logging import get_logger
from accelerate.utils import set_seed
from copy import deepcopy
# from omegaconf import OmegaConf
from utils.pc_utils import random_sampling, rotx, roty, rotz
import pickle
import MinkowskiEngine as ME
from icecream import ic
from models.fuyu_3d import Fuyu3DCausalLMv2
from collections import OrderedDict
import pretty_errors
import uuid
from fuyu_utils import (
    get_optimizer_param_groups_by_names_dict, 
    random_sampling, 
    rotx, 
    roty, 
    rotz, 
    VisualInstructionTuningDataset3D, 
    ScanReferDataset,
    Scan2CapSimpleDataset,
    ScanQADatasetUnified,
    Scan2ObjectNameDataset,
    Scan2CapTestDataset,
    OpenEndedQADataset,
    SceneCaptionDataset,
    ScanNetFrameCaptionDataset,
    get_3d_box,
    acc_iou,
    batch_parse_grounding_text,
    batch_parse_get_iou,
    score_captions,
    MergedDataset,
    metrics_qa,
    compute_qa_score,
    get_output_embedding,
    get_word_embedding,
    preprocess_sos_eos_for_scan2cap,
    postprocess_punctuation_for_caption_metrics,
    dummy_image_getter,
    FrameCaptionGetter,
)
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from typing import List, Set, Tuple, Dict, Union, Any, Optional

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
LABEL_START_TOKEN = ""
LABEL_SHIFT = None

wandb_project = "Kuri3D-merged-qa"
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

GLOBAL_CONF = None
# logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

def parse_args():
    parser = ArgumentParser()
    # Data
    # parser.add_argument("--prompt", default="Answer the following VQAv2 question based on the image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    # parser.add_argument("--i2t", type=str, default="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_view_map_new.json")
    parser.add_argument("--i2t_scanqa", type=str, default="/scratch/mowentao/BLIP/scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json")
    # parser.add_argument("--i2t_sqa3d", type=str, default="/scratch/generalvision/ScanQA-feature/scene_view_map_video.json")
    parser.add_argument("--i2t_sqa3d", type=str, default="/scratch/mowentao/BLIP/scene_eval_sqa_video_qonly.pkl")
    parser.add_argument("--i2t_scan2cap", type=str, default="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_view_map_full.json")
    parser.add_argument("--i2t_scan2cap_val", type=str, default="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_view_map_for_valtest_mask3d.json")

    parser.add_argument("--frame_path_scanqa", type=str, default="/scratch/generalvision/ScanQA-feature/frames_square/")
    parser.add_argument("--frame_path_sqa3d", type=str, default="/scratch/generalvision/ScanQA-feature/frames_square/")
    # parser.add_argument("--frame_path_sqa3d", type=str, default="/scratch/generalvision/ScanQA-feature/selected_images/")
    parser.add_argument("--frame_path_scan2cap", type=str, default="/scratch/generalvision/ScanQA-feature/frames_square/")
    # parser.add_argument("--dataset", type=str, default="scanrefer")
    # parser.add_argument("--sqa_prompt", type=str, default="Answer the following SQA3D question based on the situation and image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    parser.add_argument("--use_augment", action="store_true")
    parser.add_argument("--finetune_epochs", type=int, default=0)
    
    # parser.add_argument("--add_scanqa", action="store_true")
    # parser.add_argument("--add_scan2obj", action="store_true")
    # parser.add_argument("--add_nr3d", action="store_true")
    # parser.add_argument("--add_sr3d", action="store_true")
    # parser.add_argument("--add_scan2cap", action="store_true")
    # parser.add_argument("--add_sqa3d", action="store_true")
    # parser.add_argument("--add_lamm3d", action="store_true")
    # parser.add_argument("--add_scenecap", action="store_true")
    # parser.add_argument("--add_framecap", action="store_true")
    # parser.add_argument("--framecap_percentile", type=str, default="30.0")
    # parser.add_argument("--framecap_name", type=str, default="framecap")
    parser.add_argument("--scan2cap_predicted_bbox_file", type=str, default="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_info_for_valtest_mask3d.pkl")
    # parser.add_argument("--add_nr3d_val", action="store_true")
    
    # parser.add_argument("--add_scan2obj_val", action="store_true")
    parser.add_argument("--predict_dataset", type=str, default="scanqa,sqa3d")

    parser.add_argument("--scan2cap_metric_type", type=str, default="recall")
    parser.add_argument("--deduplicate_captions", action="store_true")
    parser.add_argument("--merged_scan2obj", action="store_true")

    # LLM or LVLM
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    
    # Modality/Prompt
    parser.add_argument("--use_dummy_image", action="store_true")
    parser.add_argument("--sample_with_sqrt_freq", action="store_true")
    parser.add_argument("--use_no_location_text", action="store_true")
    parser.add_argument("--use_dummy_image_for_scan2cap", action="store_true")
    parser.add_argument("--label_start_token", type=str, default="\x04")
    parser.add_argument("--prompt_end_token", type=str, default="|ENDOFTEXT|")
    parser.add_argument("--use_no_dataset_name", action="store_true")
    parser.add_argument("--framecap_as_input", action="store_true")
    parser.add_argument("--scale_bbox", type=int, default=100)
    
    # Optimization
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_3d", type=float, default=5e-5)
    parser.add_argument("--lr_adapter", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--weight_decay_adapter", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--best_criteria", type=str, default="em")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--unfreeze_word_embedding", action="store_true")
    parser.add_argument("--num_think_tokens", type=int, default=0)
    parser.add_argument("--use_focus_bbox", action="store_true")

    # Logging
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpointing_steps", type=float, default=5000)
    parser.add_argument("--print_log_step", type=int, default=20)
    parser.add_argument("--tag", type=str, default="")


    # 3D options
    parser.add_argument("--use_3d", action="store_true")
    parser.add_argument("--spatial_patch_size", type=int, default=24)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--pooling_method", type=str, default="max")
    parser.add_argument("--shift_bbox_to_positive", action="store_true")
    parser.add_argument("--detector_from_scratch", action="store_true")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--use_height", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--adapter_type", type=str, default="ffn")
    parser.add_argument("--num_points", type=int, default=40_000)
    parser.add_argument("--pc_tokenizer_type", type=str, default="minkowski")
    parser.add_argument("--frozen_object_feature_path", type=str, default="/scratch/generalvision/mowentao/ScanQA/data/scannetv2-pnpp-feature.pkl")
    parser.add_argument("--frozen_object_type", type=str, default="pnpp")
    parser.add_argument("--use_frozen_object_feature", action="store_true")
    parser.add_argument("--vote2cap_return_type", type=str, default="enc_features") # alt: box_features
    parser.add_argument("--no_bbox_mask_for_framecap", action="store_true")
    parser.add_argument("--predict_frame_params", action="store_true")
    parser.add_argument("--coeff_frame_params", type=float, default=0.1)

    parser.add_argument("--not_use_2d", action="store_true")
    parser.add_argument("--not_use_3d", action="store_true")
    
    #  |- Qformer
    parser.add_argument("--num_query_tokens", type=int, default=128)
    parser.add_argument("--qformer_num_hidden_layers", type=int, default=12)
    parser.add_argument("--pretrained_qformer", type=str, default="CH3COOK/bert-base-embedding")
    parser.add_argument("--use_pretrained_qformer", action="store_true")

    # LORA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    parser.add_argument("--lora_rank_finetune", type=int, default=32)
    parser.add_argument("--lora_alpha_finetune", type=float, default=64)
    parser.add_argument("--lora_dropout_finetune", type=float, default=0.05)

    parser.add_argument("--use_pissa", action="store_true")

    # External config
    parser.add_argument("--prompt_config", type=str, default="")

    # Generation/Decode options
    parser.add_argument("--generation_method", type=str, default="greedy")
    # beam search
    parser.add_argument("--num_beams", type=int, default=5)
    # nucleus sampling
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    # limit answer vocab
    parser.add_argument("--restrict_vocab", action="store_true")

    # Finetune
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--trainable_lora_in_finetune", action="store_true")
    parser.add_argument("--create_new_lora_for_finetune", action="store_true")
    parser.add_argument("--validate_at_start", action="store_true")
    parser.add_argument("--special_finetune_prompt", type=str, default="")
    parser.add_argument("--only_load_adapter", action="store_true")

    return parser.parse_args()

def get_model(args):
    # load model. NOTE: in bfloat16
    in_channels = 128 * int(args.use_multiview) + int(args.use_height) + 3 * int(args.use_normal) + 3 * int(args.use_color)
    if args.pc_tokenizer_type == "minkowski":
        in_channels += 3 # concat xyz

    logger.info(f"Using {in_channels} channels for 3D data.")

    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token # |ENDOFTEXT|
    model = Fuyu3DCausalLMv2(
        pretrained_args={
            "pretrained_model_name_or_path": model_id,
            "torch_dtype": torch.bfloat16,
        },
        mnet_path="/scratch/generalvision/mowentao/ScanQA/weights.pth",
        pnpp_path="/scratch/generalvision/mowentao/SQA3D/ScanQA/outputs/2023-06-10_00-11-47_AUXI/model_last.pth",
        vote2cap_detr_path="/scratch/generalvision/mowentao/ScanQA/LL3DA-main/pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth",
        freeze_mnet=args.lr_3d <= 1e-8,
        freeze_pnpp=args.lr_3d <= 1e-8,
        freeze_vote2cap_detr=args.lr_3d <= 1e-8,
        spatial_patch_size=args.spatial_patch_size,
        pooling_method=args.pooling_method,
        in_channels=in_channels,
        num_think_tokens=args.num_think_tokens,
        use_focus_bbox=args.use_focus_bbox,
        adapter_type=args.adapter_type,
        pc_tokenizer_type=args.pc_tokenizer_type,
        num_query_tokens=args.num_query_tokens,
        qformer_num_hidden_layers=args.qformer_num_hidden_layers,
        pretrained_qformer=args.pretrained_qformer if args.use_pretrained_qformer else None,
        vote2cap_return_type=args.vote2cap_return_type,
        use_2d=not args.not_use_2d,
        use_3d=not args.not_use_3d,
        predict_frame_params=args.predict_frame_params,
        coeff_frame_params=args.coeff_frame_params,
        frozen_in_channels=args.frozen_in_channels,
        merged_frozen_in_channels=args.merged_frozen_in_channels,
    )


    if not args.detector_from_scratch:
        model.load_detector()

    if args.checkpoint_path != "":
        logger.info(f"Loading checkpoint from {args.checkpoint_path}...")
        model.load_pretrained(args.checkpoint_path)
    return model, processor 

def get_peft_fuyu(model, args):
    if args.lora_rank == 0:
        # freeze all parameters
        logger.info(f"No LoRA applied as lora_rank == {args.lora_rank}.")
        logger.info("Freezing all parameters...")
        for p in model.fuyu.parameters():
            p.requires_grad = False
        return model

    modules_to_save = ["mnet", "linear_3d", "linear_focus_bbox", "think_tokens"]
    if args.unfreeze_word_embedding:
        modules_to_save.extend([
            "language_model.model.embed_tokens",
            "language_model.lm_head",
        ])

    
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
        modules_to_save=modules_to_save,
        init_lora_weights="pissa_niter_48" if args.use_pissa else True,
    )

    # apply LORA to the LLM
    if getattr(model, "fuyu", None) is not None:
        
        
        # else:
        logger.info(f"Applying LoRA in {type(model)} submodel Fuyu...")
        model.fuyu: PeftModel = get_peft_model(model.fuyu, peft_config)

        # load checkpoint
        if args.checkpoint_path != "" and not args.only_load_adapter:
            logger.info(f"Loading LoRA checkpoint from {args.checkpoint_path}...")
            # model.load_pretrained(args.checkpoint_path)
            # from peft import AutoPeftModel, AutoPeftModelForCausalLM
            # model.fuyu = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint_path, is_trainable=args.trainable_lora_in_finetune)
            message = model.fuyu.load_adapter(
                model_id=args.checkpoint_path,
                adapter_name="default",
                is_trainable=args.trainable_lora_in_finetune,
            )
            logger.info(message)

            if args.create_new_lora_for_finetune:
                # logger.info(f"Merging adapters...")
                model.fuyu = model.fuyu.merge_and_unload(progressbar=True)
                
                # re-apply new LoRA
                logger.info(f"Re-applying new LoRA in {type(model)} submodel Fuyu...")
                new_peft_config = deepcopy(peft_config)
                new_peft_config.r = args.lora_rank_finetune
                new_peft_config.lora_alpha = args.lora_alpha_finetune
                new_peft_config.lora_dropout = args.lora_dropout_finetune
                model.fuyu = get_peft_model(model.fuyu, new_peft_config)
                logger.info(model.fuyu.peft_config)

        
        print("Trainable parameters in LVLM:")
        model.fuyu.print_trainable_parameters()
        
        # print("Trainable parameters outside LVLM:")
        num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad and "fuyu" not in name and "llm" not in name:
                # print(name, param.numel())
                num_params += param.numel()

        print(f"Trainable parameters outside LVLM: {num_params:,d}")
    else:
        logger.info(f"Applying LoRA in {type(model)}...")
        model = get_peft_model(model, peft_config) # FIXME: will this apply LoRA to 3D encoder?
        model.print_trainable_parameters()


    return model


def get_answer_vocab(qa_datasets):
    answer_vocab: Set[str] = set()
    for dataset in qa_datasets:
        answer_vocab.update(dataset.answer_vocab)
    print(f"Total {len(answer_vocab)} answers in the merged dataset.")
    return sorted(list(answer_vocab))

def batch_generate_v2(model: Fuyu3DCausalLMv2, model_inputs, max_new_tokens=80, return_text=True, skip_special_tokens=False, generation_config={}):
    # model_inputs = model_inputs.to('cuda')

    generated = model.generate( **model_inputs, **generation_config, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id, synced_gpus=True)[:, -max_new_tokens:]

    # print(processor.batch_decode(generated, skip_special_tokens=False))
    model_outputs: List[str] = processor.tokenizer.batch_decode(generated, skip_special_tokens=skip_special_tokens)
    # print(model_outputs)
    # prediction = [m.split('\x04', 1)[1].strip() if '\x04' in m else m for m in model_outputs]
    if any(['\x04' in m for m in model_outputs]):
        prediction = [m.split('\x04', 1)[1].strip() if '\x04' in m else m for m in model_outputs]
    elif any(['[/INST]' in m for m in model_outputs]):
        prediction = [m.split('[/INST]', 1)[1].strip() if '[/INST]' in m else m for m in model_outputs]
    else:
        prediction = model_outputs

    if return_text:
        return prediction
    else:
        return {
            "model_outputs": generated,
            "prediction": prediction
        }

    
# def batch_forward(model, processor, questions, answers, images, prompt="Answer the following VQAv2 question based on the image:{}\x04 {}", max_new_tokens=80, return_text=True):
#     # model_inputs = processor(**inputs).to('cuda')
#     text = [prompt.format(q, a) for q, a in zip(questions, answers)]
#     print(text)
#     model_inputs = processor(text=text, images=images).to('cuda')
#     generated = model(**model_inputs)
    
#     return generated


def get_image_for_question(frame_path, scene_to_image, whole_qid):
    scene_id = whole_qid.split("-")[1] # train-xxxx-xxxx
    image = scene_to_image[whole_qid][0] # xxxx.jpg
    image_path = os.path.join(frame_path, f"{scene_id}_00/color/{image}")
    return Image.open(image_path)

def get_test_datasets(args) -> dict[str, Dataset]:
    shared_config = {
        "ratio": args.train_ratio,
        "use_color": args.use_color,
        "use_multiview": args.use_multiview,
        "use_height": args.use_height,
        "use_normal": args.use_normal,
        "shift_bbox_to_positive": args.shift_bbox_to_positive,
        "num_points": args.num_points,
        "prompt_end_token": args.prompt_end_token,
        "prompt": [args.special_finetune_prompt] if len(args.special_finetune_prompt) > 0 else None,
        "framecap_as_input": args.framecap_as_input,
        "scale_bbox": args.scale_bbox,
        "use_llm_style_prompt": args.use_llm,
        "frozen_object_type": args.frozen_object_type,
        # "frozen_object_feature_path": args.frozen_object_feature_path if args.use_frozen_object_feature else None,
    }

    shared_densecap_config = {
        "use_no_location_text": args.use_no_location_text,
        "use_no_dataset_name": args.use_no_dataset_name,
    }

    to_predict_datasets = args.predict_dataset.split(",")
    logger.info(f"Predicting on datasets: {to_predict_datasets}")

    datasets = {}

    # val-scanqa
    if "scanqa" in to_predict_datasets:
        val_dataset = ScanQADatasetUnified(
            name="scanqa",
            split="test_w_obj",
            use_augment=False,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            **shared_config,
        )
        datasets["scanqa-test_w_obj"] = val_dataset

        val_dataset = ScanQADatasetUnified(
            name="scanqa",
            split="test_wo_obj",
            use_augment=False,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            **shared_config,
        )
        datasets["scanqa-test_wo_obj"] = val_dataset

    # val-sqa3d
    if "sqa3d" in to_predict_datasets:
        val_dataset_sqa3d = ScanQADatasetUnified(
            name="sqa3d",
            split="val",
            use_augment=False,
            i2t=args.i2t_sqa3d,
            views_path=args.frame_path_sqa3d,
            **shared_config,
        )
        test_dataset_sqa3d = ScanQADatasetUnified(
            name="sqa3d",
            split="test",
            use_augment=False,
            i2t=args.i2t_sqa3d,
            views_path=args.frame_path_sqa3d,
            **shared_config,
        )
        datasets["sqa3d-val"] = val_dataset_sqa3d
        datasets["sqa3d-test"] = test_dataset_sqa3d

    # scenecap
    if "scenecap" in to_predict_datasets:
        scenecap_dataset = SceneCaptionDataset(
            name="scenecap",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap, # dummy
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
        datasets["scenecap-val"] = scenecap_dataset

    # scan2obj
    if "scan2obj" in to_predict_datasets:
        scan2obj_dataset = Scan2ObjectNameDataset(
            name="scan2obj",
            split="val",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            use_no_location_text=args.use_no_location_text,
            **shared_config,
        )
        # if args.deduplicate_captions:
        scan2obj_dataset.deduplicate_captions() # object name is a must to deduplicate
        datasets["scan2obj-val"] = scan2obj_dataset

    def recursive_set_tokenizer_type(dataset_or_list, pc_tokenizer_type):
        if isinstance(dataset_or_list, list):
            for d in dataset_or_list:
                recursive_set_tokenizer_type(d, pc_tokenizer_type)
        elif isinstance(dataset_or_list, MergedDataset):
            for d in dataset_or_list.datasets:
                d.set_pc_tokenizer_type(pc_tokenizer_type)
        elif dataset_or_list is not None:
            dataset_or_list.set_pc_tokenizer_type(pc_tokenizer_type)

    recursive_set_tokenizer_type(list(datasets.values()), args.pc_tokenizer_type)

    # set frozen_in_channels
    args.frozen_in_channels = getattr(datasets[0], "frozen_in_channels", 1)
    args.merged_frozen_in_channels = getattr(datasets[0], "merged_frozen_in_channels", [256,256])
    # args.merged_frozen_in_channels = args.merged_frozen_in_channels[]

    return datasets




def get_trainval_datasets(args) -> dict[str, Dataset]:
    if args.special_finetune_prompt != "":
        args.special_finetune_prompt = args.special_finetune_prompt.encode().decode('unicode_escape')
        logger.info(f"Using special finetune prompt: {repr(args.special_finetune_prompt)}")

    shared_config = {
        "ratio": args.train_ratio,
        "use_color": args.use_color,
        "use_multiview": args.use_multiview,
        "use_height": args.use_height,
        "use_normal": args.use_normal,
        "shift_bbox_to_positive": args.shift_bbox_to_positive,
        "num_points": args.num_points,
        "prompt_end_token": args.prompt_end_token,
        "prompt": [args.special_finetune_prompt] if len(args.special_finetune_prompt) > 0 else None,
        "framecap_as_input": args.framecap_as_input,
        "scale_bbox": args.scale_bbox,
        "use_llm_style_prompt": args.use_llm,
        "frozen_object_type": args.frozen_object_type,
        # "frozen_object_feature_path": args.frozen_object_feature_path if args.use_frozen_object_feature else None,
    }

    shared_densecap_config = {
        "use_no_location_text": args.use_no_location_text,
        "use_no_dataset_name": args.use_no_dataset_name,
    }

    # make merged dataset: ScanQA + SQA3D + Scan2Cap (on ScanRefer)
    datasets = []

    if args.add_scanqa:
        scanqa_train = ScanQADatasetUnified(
            name="scanqa",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            **shared_config,
        )
        datasets.append(scanqa_train)

    if args.add_sqa3d:
        sqa3d_train = ScanQADatasetUnified(
            name="sqa3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_sqa3d,
            views_path=args.frame_path_sqa3d,
            **shared_config,
        )
        datasets.append(sqa3d_train)

    if args.add_lamm3d:
        lamm3d = OpenEndedQADataset(
            name="lamm3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            **shared_config,
        )
        datasets.append(lamm3d)
    
    if args.add_scan2cap:
        scan2cap_dataset = Scan2CapSimpleDataset(
            name="scan2cap",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            **shared_densecap_config,
            **shared_config,
        )
        if args.deduplicate_captions:
            scan2cap_dataset.deduplicate_captions()
        datasets.append(scan2cap_dataset)


    if args.add_scan2obj:
        scan2obj_dataset = Scan2ObjectNameDataset(
            name="scan2obj",
            split="train" if not args.merged_scan2obj else "merged",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            use_no_location_text=args.use_no_location_text,
            **shared_config,
        )
        # if args.deduplicate_captions:
        scan2obj_dataset.deduplicate_captions() # object name is a must to deduplicate
        datasets.append(scan2obj_dataset)

    if args.add_nr3d:
        scan2cap_nr3d_dataset = Scan2CapSimpleDataset(
            name="scan2cap-nr3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            # use_no_location_text=args.use_no_location_text,
            **shared_densecap_config,
            **shared_config,
        )
        if args.deduplicate_captions:
            scan2cap_nr3d_dataset.deduplicate_captions()
        datasets.append(scan2cap_nr3d_dataset)

    if args.add_sr3d:
        scan2cap_sr3d_dataset = Scan2CapSimpleDataset(
            name="scan2cap-sr3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            # use_no_location_text=args.use_no_location_text,
            **shared_densecap_config,
            **shared_config,
        )
        if args.deduplicate_captions:
            scan2cap_sr3d_dataset.deduplicate_captions()
        datasets.append(scan2cap_sr3d_dataset)

    if args.add_scenecap:
        scenecap_dataset = SceneCaptionDataset(
            name="scenecap",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap, # dummy
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
        datasets.append(scenecap_dataset)

    if args.add_framecap:
        framecap_dataset = ScanNetFrameCaptionDataset(
            name="framecap",
            dataset_type=args.framecap_name,
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap, # dummy
            views_path=args.frame_path_scan2cap,
            percentile=args.framecap_percentile,
            **shared_config,
        )
        datasets.append(framecap_dataset)

        if args.framecap_as_input:
            val_framecap_dataset = ScanNetFrameCaptionDataset(
                name="framecap",
                dataset_type=args.framecap_name,
                split="val",
                use_augment=args.use_augment,
                i2t=args.i2t_scan2cap, # dummy
                views_path=args.frame_path_scan2cap,
                percentile=args.framecap_percentile,
                **shared_config,
            )
            test_framecap_dataset = ScanNetFrameCaptionDataset(
                name="framecap",
                dataset_type=args.framecap_name,
                split="test",
                use_augment=args.use_augment,
                i2t=args.i2t_scan2cap, # dummy
                views_path=args.frame_path_scan2cap,
                percentile=args.framecap_percentile,
                **shared_config,
            )

            FrameCaptionGetter().setup_captions_from_annotation([
                framecap_dataset.annotation,
                val_framecap_dataset.annotation,
                test_framecap_dataset.annotation,
            ]) # gather all scene's frame captions
    
    if args.use_dummy_image:
        for dataset in datasets:
            dataset.image_getter = dummy_image_getter

    if args.use_dummy_image_for_scan2cap:
        if scan2cap_dataset is not None:
            scan2cap_dataset.image_getter = dummy_image_getter
        if scan2obj_dataset is not None:
            scan2obj_dataset.image_getter = dummy_image_getter
        if scan2cap_nr3d_dataset is not None:
            scan2cap_nr3d_dataset.image_getter = dummy_image_getter
        if scan2cap_sr3d_dataset is not None:
            scan2cap_sr3d_dataset.image_getter = dummy_image_getter

    

    # merge datasets
    train_dataset = MergedDataset(
        datasets=datasets,
        sample_with_sqrt_freq=args.sample_with_sqrt_freq,
    )

    # for dataset in datasets:
    #     dataset.set_pc_tokenizer_type(args.pc_tokenizer_type)

    # val-scanqa
    if args.add_scanqa:
        val_dataset = ScanQADatasetUnified(
            name="scanqa",
            split="val",
            use_augment=False,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            **shared_config,
        )
    else:
        val_dataset = None

    # val-sqa3d
    if args.add_sqa3d:
        val_dataset_sqa3d = ScanQADatasetUnified(
            name="sqa3d",
            split="val",
            use_augment=False,
            i2t=args.i2t_sqa3d,
            views_path=args.frame_path_sqa3d,
            **shared_config,
        )
        test_dataset_sqa3d = ScanQADatasetUnified(
            name="sqa3d",
            split="test",
            use_augment=False,
            i2t=args.i2t_sqa3d,
            views_path=args.frame_path_sqa3d,
            **shared_config,
        )
    else:
        val_dataset_sqa3d = None
        test_dataset_sqa3d = None

    # val-nr3d
    if args.add_nr3d_val:
        val_dataset_nr3d = Scan2CapTestDataset(
            name="scan2cap-nr3d",
            split="val",
            use_augment=False,
            # i2t=args.i2t_scan2cap,
            i2t=args.i2t_scan2cap_val,
            views_path=args.frame_path_scan2cap,
            bbox_file=args.scan2cap_predicted_bbox_file,
            # use_no_location_text=args.use_no_location_text,
            **shared_densecap_config,
            **shared_config,
        )
    else:
        val_dataset_nr3d = None

    # val-scan2cap
    if args.add_scan2cap:
        val_dataset_scan2cap = Scan2CapTestDataset(
            name="scan2cap",
            split="val",
            use_augment=False,
            # i2t=args.i2t_scan2cap,
            i2t=args.i2t_scan2cap_val,
            views_path=args.frame_path_scan2cap,
            bbox_file=args.scan2cap_predicted_bbox_file,
            # use_no_location_text=args.use_no_location_text,
            **shared_densecap_config,
            **shared_config,
        )
        # val_dataset_scan2cap_gt = Scan2CapSimpleDataset(
        #     name="scan2cap",
        #     split="val",
        #     use_augment=False,
        #     # i2t=args.i2t_scan2cap,
        #     i2t=args.i2t_scan2cap,
        #     views_path=args.frame_path_scan2cap,
        #     use_no_location_text=args.use_no_location_text,
        #     **shared_config,
        # )
        val_dataset_scan2cap_gt = None

    else:
        val_dataset_scan2cap = None
        val_dataset_scan2cap_gt = None

    # if args.add_scan2obj_val:
    #     val_dataset_scan2obj = Scan2ObjectNameDataset(
    #         name="scan2obj",
    #         split="val",
    #         use_augment=False,
    #         i2t=args.i2t_scan2cap,
    #         views_path=args.frame_path_scan2cap,
    #         use_no_location_text=args.use_no_location_text,
    #         **shared_config,
    #     )
    #     val_dataset_scan2obj.deduplicate_captions()
    
    if args.use_dummy_image_for_scan2cap:
        val_dataset_scan2cap.image_getter = dummy_image_getter
        val_dataset_scan2cap_gt.image_getter = dummy_image_getter
        val_dataset_nr3d.image_getter = dummy_image_getter

    # for dataset in [val_dataset, val_dataset_sqa3d, val_dataset_scan2cap, test_dataset_sqa3d, val_dataset_scan2cap_gt, val_dataset_nr3d]:
    #     if dataset is not None:
    #         dataset.set_pc_tokenizer_type(args.pc_tokenizer_type)

    # answer vocab
    if args.add_scanqa:
        answer_vocab = get_answer_vocab([scanqa_train, val_dataset]) # for scanqa eval
        args.answer_vocab = answer_vocab
    
    dataset_dict = {
        # "finetune": scanqa_train,
        "finetune": datasets[0], # FIXME: make it any dataset
        "train": train_dataset,
        "val": val_dataset,
        "val-sqa3d": val_dataset_sqa3d,
        "val-scan2cap": val_dataset_scan2cap,
        "test-sqa3d": test_dataset_sqa3d,
        "val-scan2cap-gt": val_dataset_scan2cap_gt,
        "val-nr3d": val_dataset_nr3d,
    }

    def recursive_set_tokenizer_type(dataset_or_list, pc_tokenizer_type):
        if isinstance(dataset_or_list, list):
            for d in dataset_or_list:
                recursive_set_tokenizer_type(d, pc_tokenizer_type)
        elif isinstance(dataset_or_list, MergedDataset):
            for d in dataset_or_list.datasets:
                d.set_pc_tokenizer_type(pc_tokenizer_type)
        elif dataset_or_list is not None:
            dataset_or_list.set_pc_tokenizer_type(pc_tokenizer_type)

    recursive_set_tokenizer_type(list(dataset_dict.values()), args.pc_tokenizer_type)

    # set frozen_in_channels
    args.frozen_in_channels = getattr(datasets[0], "frozen_in_channels", 1)
    args.merged_frozen_in_channels = getattr(datasets[0], "merged_frozen_in_channels", [256,256])
    # args.merged_frozen_in_channels = args.merged_frozen_in_channels[]

    return dataset_dict



def collate_3d(examples):
    # minkowski
    if "coords" in examples[0]:
        data_3d = [(e["coords"], e["feats"], e["labels"]) for e in examples]
        data_3d = ME.utils.batch_sparse_collate(data_3d)

        return data_3d

    # pointnet++ or vote2cap-detr
    elif "point_cloud" in examples[0]:
        point_cloud = np.stack([e["point_cloud"] for e in examples], axis=0)
        if "point_cloud_dims_min" in examples[0]: # vote2cap-detr
            point_cloud_dims_min = np.stack([e["point_cloud_dims_min"] for e in examples], axis=0)
            point_cloud_dims_max = np.stack([e["point_cloud_dims_max"] for e in examples], axis=0)
            return [
                torch.tensor(point_cloud, dtype=torch.float32),
                torch.tensor(point_cloud_dims_min, dtype=torch.float32),
                torch.tensor(point_cloud_dims_max, dtype=torch.float32),
            ]
            
        return torch.tensor(point_cloud, dtype=torch.float32)

    # frozen
    elif "object_feature" in examples[0]:
        # merged-frozen
        if examples[0].get("merged-frozen", False):
            # e["object_feature"] is a list of features
            object_feature = [
                torch.from_numpy(np.stack([e["object_feature"][i] for e in examples], axis=0)).float()
                for i in range(len(examples[0]["object_feature"]))
            ]

            object_mask = [
                torch.from_numpy(np.stack([e["object_mask"][i] for e in examples], axis=0)).bool()
                for i in range(len(examples[0]["object_mask"]))
            ]

            data = [object_feature, object_mask]

            if "predicted_bbox_corners" in examples[0]:
                predicted_bbox_corners = [
                    torch.from_numpy(np.stack([e["predicted_bbox_corners"][i] for e in examples], axis=0)).float()
                    for i in range(len(examples[0]["predicted_bbox_corners"]))
                ]
                data.append(predicted_bbox_corners)

            return data
        else:
            object_feature = torch.from_numpy(np.stack([e["object_feature"] for e in examples], axis=0)).float()
            object_mask = torch.from_numpy(np.stack([e["object_mask"] for e in examples], axis=0)).bool()
            data = [object_feature, object_mask]
            if "predicted_bbox_corners" in examples[0]:
                predicted_bbox_corners = torch.from_numpy(np.stack([e["predicted_bbox_corners"] for e in examples], axis=0)).float()
                data.append(predicted_bbox_corners)

            return data
    
    raise NotImplementedError("Unknown 3D data format with keys: " + str(examples[0].keys()))

def collate_vl(examples) -> dict:
    global processor, qtokenizer
    texts = [e['target'] for e in examples]
    # print(texts)
    images = [e["image"] for e in examples]
    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True
    )

    texts_instructions = [e['target_instruction'] for e in examples] # without \x04 and later tokens (gt annotations)
    qformer_inputs = qtokenizer(texts_instructions, padding="longest", return_tensors="pt")
    output["qformer_inputs"] = qformer_inputs

    return output

def collate_frame_info(output, examples) -> dict:
    output["frame_intrinsics"] = torch.tensor(np.stack([e["frame_intrinsics"] for e in examples])).to(torch.float32)
    output["frame_poses"] = torch.tensor(np.stack([e["frame_poses"] for e in examples])).to(torch.float32)
    output["axis_alignments"] = torch.tensor(np.stack([e["axis_alignments"] for e in examples])).to(torch.float32)
    output["frame_caption_mask"] = torch.tensor(np.stack([e["frame_caption_mask"] for e in examples])).to(torch.bool)
    if GLOBAL_CONF.no_bbox_mask_for_framecap:
        output["frame_caption_mask"] = torch.zeros_like(output["frame_caption_mask"]) # mark as no sample is from frame caption

    return output

def collate_fn(examples) -> dict:
    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    output["focus_bbox"] = torch.tensor(np.stack([e["target_bbox"] for e in examples]))
    has_focus_bbox_types = ["scan2cap", "scan2obj", "scan2cap-nr3d", "scan2cap-sr3d"]
    has_focus_bbox_masks = [e["data_type"] in has_focus_bbox_types for e in examples]
    output["focus_bbox_mask"] = torch.tensor(has_focus_bbox_masks)

    positions = [] # label start position (before \x04)

    # x04_token = processor.tokenizer("\x04", add_special_tokens=False)["input_ids"][-1] # first is <s>
    # start_token = processor.tokenizer(LABEL_START_TOKEN, add_special_tokens=False)["input_ids"][-1] # first is <s>
    # start_token = LABEL_START_TOKEN
    if LABEL_START_TOKEN == "\x04":
        start_token = processor.tokenizer("\x04", add_special_tokens=False)["input_ids"][-1]
    elif LABEL_START_TOKEN == "[/INST]":
        start_token = processor.tokenizer("[/INST]", add_special_tokens=False)["input_ids"][-2] 
    for input_ids in output["input_ids"]:
        # position = (input_ids == processor.tokenizer.vocab["<s>"]).nonzero(as_tuple=True)[0][0]
        if LABEL_START_TOKEN == "\x04":
            position = (input_ids == start_token).nonzero(as_tuple=True)[0][0]
        elif LABEL_START_TOKEN == "[/INST]":
            position = (input_ids == start_token).nonzero(as_tuple=True)[0][-1] + 1
            # print(position)

        positions.append(position - 1) # -1 to consider that \x04? or +2 to not consider \x04 and the space after it
            # and for "INST" in "[/INST]", the label starts after the "]"
    # positions = torch.tensor(positions)
    output["labels"] = torch.full_like(output["input_ids"], -100)  # This creates a tensor filled with -100
    for i in range(len(positions)):
        output["labels"][i, positions[i]:] = output["input_ids"][i, positions[i]:] # mask the tokens before <s> in labels
    # print(output["labels"])

    output["mnet_inputs"] = collate_3d(examples)

    return output

def collate_fn_eval(examples) -> dict:
    to_copy_keys = ["raw_question_id", "answers"]

    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    for k in to_copy_keys:
        output[k] = [e[k] for e in examples]
    output["to_copy_keys"] = to_copy_keys
    output["mnet_inputs"] = collate_3d(examples)
    
    return output

def collate_fn_eval_scan2cap(examples) -> dict:
    to_copy_keys = ["raw_question_id", "question_id", "target", "scan2cap_id", "description"]

    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    output["focus_bbox"] = torch.tensor(np.stack([e["target_bbox"] for e in examples]))

    for k in to_copy_keys:
        output[k] = [e[k] for e in examples]
    output["to_copy_keys"] = to_copy_keys
    output["mnet_inputs"] = collate_3d(examples)
    
    return output


def get_trainval_dataloaders(args) -> dict[str, DataLoader]:
    def get_dataloader(dataset, split):
        if dataset is None:
            return None

        is_train_dataset = split in ["train", "finetune"]
        if is_train_dataset:
            collate_fn_for_dataset = collate_fn
        else:
            collate_fn_for_dataset = (
                collate_fn_eval if split in [
                    "scanqa-val",
                    "scanqa-test_w_obj",
                    "scanqa-test_wo_obj",
                    "sqa3d-val",
                    "sqa3d-test",
                    "scenecap-val",
                    "scan2obj-val",
                    ]
                else collate_fn_eval_scan2cap
            )

        return DataLoader(
            dataset, 
            batch_size=args.batch_size if is_train_dataset else args.eval_batch_size,
            shuffle=is_train_dataset, 
            collate_fn=collate_fn_for_dataset,
            num_workers=args.num_workers,
        )

    dataloaders = {
        split: get_dataloader(dataset, split) for split, dataset in args.datasets.items()
    }

    return dataloaders

def get_optimizer_scheduler(args, model):
    # optimizer
    name_to_params_group_map = OrderedDict()
    name_to_params_group_map["params_adapter"] = ["linear_3d", "linear_focus_bbox", "think_tokens", "qformer"]
    name_to_params_group_map["params_3d"] = ["pc_tokenizer"]
    name_to_params_group_map["no_decay"] = ["bias", "layer_norm.weight"]
    # other params including the LVLM (fuyu)
    lr_dict = {
        "params_3d": args.lr_3d,
        "params_adapter": args.lr_adapter,
        "no_decay": args.lr,
    }
    weight_decay_dict = {
        "params_3d": args.weight_decay,
        "params_adapter": args.weight_decay_adapter,
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
    logger.info(optimizer_grouped_parameter_names)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
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
    model: Fuyu3DCausalLMv2 = args.model

    if args.is_finetuning:
        dataloader = args.dataloaders["finetune"]
        logger.info(f"Finetuning for {epoch - args.finetune_epochs}/{args.finetune_epochs} epochs...")
    else:
        dataloader = args.dataloaders["train"]
        logger.info(f"Training for {epoch}/{args.epochs - args.finetune_epochs} epochs...")

    for step, batch in enumerate(dataloader):

        with accelerator.accumulate(model):
            outputs = model(return_dict=True, **batch) 
            loss = outputs.loss
            accelerator.backward(loss)
            # print(outputs, loss)

            # report loss to wandb
            # reduce loss over processes for logging purposes
            with torch.no_grad():
                loss_mean = accelerator.gather(loss.unsqueeze(0)).mean()
                args.history_losses.append(loss_mean.item())
                loss_frame = getattr(outputs, "loss_frame", None)
                if loss_frame is not None:
                    loss_frame = accelerator.gather(loss_frame.unsqueeze(0)).mean().item()
                else:
                    loss_frame = 0

                if accelerator.is_local_main_process:
                    wandb.log({
                        "train/loss": args.history_losses[-1], 
                        "train/loss_frame": loss_frame,
                        "train/global_step": len(args.history_losses),
                        "train/seen_samples": len(args.history_losses) * args.batch_size * args.accelerator.num_processes,
                        "train/lr": args.scheduler.get_last_lr()[0],
                    })

            if args.gradient_clipping > 1e-8:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            args.optimizer.step()
            args.scheduler.step()
            args.optimizer.zero_grad()

        if step % args.print_log_step == 0:
            if accelerator.is_local_main_process:
                print(f"[{epoch}/{len(args.history_losses)}-{args.total_steps}]loss: {args.history_losses[-1]}, lr: {args.scheduler.get_last_lr()[0]}, loss_frame: {loss_frame}")
                # wandb.log({
                #     # "train/loss": args.history_losses[-1], 
                #     "train/loss": np.mean(args.history_losses[-args.print_log_step:]),
                #     "train/global_step": len(args.history_losses),
                #     "train/seen_samples": len(args.history_losses) * args.batch_size * args.accelerator.num_processes,
                #     "train/lr": args.scheduler.get_last_lr()[0],
                # })

        if len(args.history_losses) % args.checkpointing_steps == 0: # validate at the first step
            # validate
            logger.info(f"[{epoch}]validating @ step {len(args.history_losses)}...")

            # save checkpoint
            # only save PEFT lora
            # only save when full training
            if args.train_ratio == 1:
                output_dir_ckpt = os.path.join(output_dir, f"ckpt-{len(args.history_losses)}")
                if not os.path.exists(output_dir_ckpt):
                    os.makedirs(output_dir_ckpt, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir_ckpt)

            validate(args, epoch)
            args.model.train()

    torch.cuda.empty_cache()

        
def validate(args, epoch):
    logger.info(f"[{epoch}]start validating...")
    args.model.eval()
    torch.cuda.empty_cache()
    accelerator: Accelerator = args.accelerator

    if is_deepspeed_zero3_enabled():
        import deepspeed
        logger.info("DeepSpeed ZeRO-3 detected. try to save model once, to make all processes have the full model.")
        # save model once, to make all processes have the full model
        unwrapped_model: Fuyu3DCausalLMv2 = accelerator.unwrap_model(args.model)
        unwrapped_model.save_pretrained("./tmp/kuri-model")

        # unwrapped_model.load_pretrained("./tmp/kuri-model")
    
    all_metrics = {}
    # correct_em, correct_refined_em, total = 0, 0, 0
    # for kind, dataloader in {
    #     "scan2cap": args.dataloaders["val-scan2cap"],
    #     "scanqa": args.dataloaders["val"],
    #     "sqa3d": args.dataloaders["val-sqa3d"],
    #     "sqa3d-test": args.dataloaders["test-sqa3d"],
    #     "scan2cap-gt": args.dataloaders["val-scan2cap-gt"],
    #     "nr3d": args.dataloaders["val-nr3d"],
    # }.items():
    for kind, dataloader in args.dataloaders.items():
        if dataloader is None:
            continue

        preds, target_texts = {}, {}
        logger.info(f"[{epoch}]validating on {kind}...")
        for step, batch in enumerate(dataloader):

            with torch.no_grad():
                to_copy_keys = batch.pop("to_copy_keys")
                copied_dict = {k: deepcopy(batch[k]) for k in to_copy_keys}
                for k in to_copy_keys:
                    del batch[k]

                # if kind in ["scanqa", "sqa3d", "sqa3d-test"]:
                    # target_text = copied_dict["answers"]
                # elif kind in ["scan2cap", "scan2cap-gt", "nr3d"]:
                    # target_text = copied_dict["description"]
                if kind in ["scanqa", "sqa3d", "scanqa-val", "sqa3d-val", "sqa3d-test", "scanqa-test_w_obj", "scanqa-test_wo_obj"]:
                    target_text = copied_dict["answers"]
                elif kind in ["scan2cap", "scan2cap-gt", "nr3d", "scenecap-val", "scan2obj-val"]:
                    target_text = copied_dict["description"]
                else:
                    raise NotImplementedError(f"kind {kind} not implemented.")
                
                # inference by generate
                unwrapped_model = accelerator.unwrap_model(args.model, keep_fp32_wrapper=True)
                if accelerator.is_local_main_process:
                    transformers.utils.logging.set_verbosity_error() # disable logging
                # print(unwrapped_model.forward, getattr(unwrapped_model.forward, "__wrapped__", None))
                pred_answer = batch_generate_v2(
                    unwrapped_model, 
                    # prompt=args.prompt.split("\x04")[0], 
                    batch,
                    max_new_tokens=args.max_new_tokens,
                    return_text=True,
                    skip_special_tokens=True,
                    generation_config=args.generation_config,
                )
                if accelerator.is_local_main_process:
                    transformers.utils.logging.set_verbosity_info() # reset to info
                if args.verbose and step % args.print_log_step == 0 :
                    logger.info(f"[{epoch}-val] {step}/{len(dataloader)}")
                    logger.info(target_text)
                    logger.info(pred_answer)

                for i, answer in enumerate(pred_answer):
                    if len(answer.strip()) == 0:
                        logger.warning(f"Empty answer detected: {answer}")
                        pred_answer[i] = "empty"

                if kind in ["scanqa", "sqa3d", "sqa3d-test", "scan2cap-gt"]:
                    for i in range(len(pred_answer)):
                        # random_uid = str(uuid.uuid4())
                        random_uid = copied_dict["raw_question_id"][i]
                        if kind in ["scan2cap-gt"]:
                            # preds[random_uid] = [preprocess_sos_eos_for_scan2cap(pred_answer[i])] # for evaluation, it shall be a list of predictions
                            preds[random_uid] = [
                                postprocess_punctuation_for_caption_metrics(
                                    preprocess_sos_eos_for_scan2cap(pred_answer[i])
                                )
                            ]
                            target_texts[random_uid] = deepcopy(args.datasets["val-scan2cap-gt"].corpus[copied_dict["scan2cap_id"][i]])
                        else:
                            preds[random_uid] = [pred_answer[i]] # for evaluation, it shall be a list of predictions
                            target_texts[random_uid] = target_text[i]
                elif kind in ["scan2cap", "nr3d"]:
                    for pred, sample_id in zip(pred_answer, copied_dict["scan2cap_id"]):
                        # preds[sample_id] = [preprocess_sos_eos_for_scan2cap(pred)] # for evaluation, it shall be a list of predictions
                        preds[sample_id] = [
                            postprocess_punctuation_for_caption_metrics(
                                preprocess_sos_eos_for_scan2cap(pred)
                            )
                        ]
                else:
                    raise NotImplementedError(f"kind {kind} not implemented.")
                
        # evaluate
        # gather all predictions and targets
        if accelerator.is_local_main_process:
            print("Gathering predictions and targets...")
        preds = gather_object([(k, v) for k, v in preds.items()])
        preds = {k: v for k, v in preds}
        if kind in ["scanqa", "sqa3d", "sqa3d-test", "scan2cap-gt"]:
            # text similarity
            target_texts = gather_object([(k, v) for k, v in target_texts.items()])
            target_texts = {k: v for k, v in target_texts}
            score_per_caption, metric_message, metrics_epoch = score_captions(target_texts, preds)

            # exact match
            em, refined_em = compute_qa_score(preds, target_texts)
            metrics_epoch["em"] = em
            metrics_epoch["refined_em"] = refined_em
        
        elif kind in ["scan2cap", "nr3d"]:
            # caption target is the corpus itself
            # score_per_caption, metric_message, metrics_epoch = score_captions(args.datasets["val-scan2cap"].corpus, preds)
            scan2cap_dataset: Scan2CapTestDataset = args.datasets["val-scan2cap"] if kind == "scan2cap" else args.datasets["val-nr3d"]
            metrics_epoch = {}
            metric_message = ""
            for iou in [0.25, 0.5]:
                gt_corpus, pred_corpus = scan2cap_dataset.process_predictions(preds, iou_threshold=iou, method=args.scan2cap_metric_type)
                # print(pred_corpus.keys(), gt_corpus.keys())
                score_per_caption, metric_message_iou, metrics_epoch_iou = score_captions(gt_corpus, pred_corpus)

                metric_message += f"Scoring captions with iou={iou}...\n"
                metric_message += metric_message_iou
                metrics_epoch.update({
                    f"{k}@{iou}": v for k, v in metrics_epoch_iou.items()
                })

        else:
            raise NotImplementedError(f"Dataset kind {kind} not implemented.")
        
        # save prediction
        if accelerator.is_local_main_process:
            logger.info("Saving predictions...")
            output_dir_pred = os.path.join(output_dir, f"ckpt-{len(args.history_losses)}", "pred")
            if not os.path.exists(output_dir_pred):
                os.makedirs(output_dir_pred, exist_ok=True)
            with open(os.path.join(output_dir_pred, f"{kind}.json"), "w") as f:
                json.dump(preds, f, indent=4)

        metrics_epoch = {
            f"{kind}_{k}": v for k, v in metrics_epoch.items()
        }
        all_metrics.update(metrics_epoch)

        if accelerator.is_local_main_process:
            print(metric_message)
    
    # for log compatibility
    if "scanqa_em" in all_metrics:
        all_metrics["em"] = all_metrics["scanqa_em"]
        all_metrics["refined_em"] = all_metrics["scanqa_refined_em"]

    if args.best_metrics.get(args.best_criteria, -1) < all_metrics[args.best_criteria]:
        args.best_metrics = all_metrics
        if accelerator.is_local_main_process:
            print(f"New best {args.best_criteria}: {args.best_metrics[args.best_criteria]}")
            # save checkpoint
            output_dir_ckpt = os.path.join(output_dir, f"best-{args.best_criteria}")
            if not os.path.exists(output_dir_ckpt):
                os.makedirs(output_dir_ckpt, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(args.model)
            unwrapped_model.save_pretrained(output_dir_ckpt)
            
    if accelerator.is_local_main_process:
        # print(f"[{epoch}]em: {metrics_epoch['em']}, refined_em: {metrics_epoch['refined_em']}")
        # print(f"[{epoch}]acc@0.5: {metrics_epoch['acc@0.5']}, acc@0.25: {metrics_epoch['acc@0.25']}")
        print(f"[{epoch}-{len(args.history_losses)}]")
        for k, v in all_metrics.items():
            print(f"[{epoch}]{k}: {v}")
        result = {
            "train/global_step": len(args.history_losses), 
            "eval/epoch": epoch,
        }
        # add current and best metrics
        result.update(all_metrics)
        result.update({
            "best/" + k: v for k, v in args.best_metrics.items()
        })
        wandb.log(result)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    GLOBAL_CONF = args

    device = args.device

    args.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    args.slurm_gpus = os.environ.get("SLURM_GPUS", None)
    
    
        

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    args.accelerator = accelerator
    
    state = accelerator.state
    if state.distributed_type == DistributedType.DEEPSPEED:
        state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.batch_size

    if accelerator.is_local_main_process:
        print(args)

    # set log options
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

    args.generation_config["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    

    accelerator.wait_for_everyone()

    datasets = get_trainval_datasets(args)
    args.datasets = datasets
    dataloaders = get_trainval_dataloaders(args)
    args.dataloaders = dataloaders

    
    model, processor = get_model(args)
    model = get_peft_fuyu(model, args)

    if args.use_llm:
        # args.label_start_token = "INST"
        # LABEL_START_TOKEN = processor.tokenizer("[/INST]", add_special_tokens=False)["input_ids"][-2] # "INST" in [ || / || INST || ]
        LABEL_START_TOKEN = "[/INST]"
        LABEL_SHIFT = 0
    else:
        # LABEL_START_TOKEN = processor.tokenizer("\x04", add_special_tokens=False)["input_ids"][-1]
        LABEL_START_TOKEN = "\x04"
        LABEL_SHIFT = -1
    # LABEL_START_TOKEN = label_start_token
    logger.info(f"LLM/LVLM LABEL_START_TOKEN: {LABEL_START_TOKEN}, LABEL_SHIFT: {LABEL_SHIFT}")

    if is_deepspeed_zero3_enabled():
        # deepspeed.utils.set_z3_leaf_modules(model, [LinearEncoders])
        pass


    args.model = model
    args.processor = processor
    
    # init instructblip tokenizer
    qtokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.pretrained_qformer)
    args.qtokenizer = qtokenizer

    logger.info(f"Total {len(args.datasets['train'])} training samples.")
    logger.info(f"Total {len(args.datasets['finetune'])} finetune samples.")
    # logger.info(f"Total {len(args.datasets['val'])} validation samples.")
    total_val_size = (
        len(args.datasets['val']) if args.datasets['val'] is not None else 0
        + len(args.datasets['val-sqa3d']) if args.datasets['val-sqa3d'] is not None else 0
        + len(args.datasets['val-scan2cap']) if args.datasets['val-scan2cap'] is not None else 0
        + len(args.datasets['test-sqa3d']) if args.datasets['test-sqa3d'] is not None else 0
    )
    logger.info(f"Total {total_val_size} validation samples.")
    # calc total steps
    # args.total_steps = args.epochs * len(args.datasets["train"]) // args.batch_size // accelerator.num_processes
    total_pretrain_steps = (args.epochs - args.finetune_epochs) * len(args.datasets["train"]) // args.batch_size // accelerator.num_processes
    total_finetune_steps = args.finetune_epochs * len(args.datasets["finetune"]) // args.batch_size // accelerator.num_processes
    args.total_steps = total_pretrain_steps + total_finetune_steps
    args.warmup_steps = min(args.total_steps // 10, 500)
    logger.info(f"Total {total_pretrain_steps} pretrain steps.")
    logger.info(f"Total {total_finetune_steps} finetune steps.")
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
    # args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"], args.dataloaders["val-sqa3d"], args.dataloaders["val-scan2cap"], args.dataloaders["finetune"] = accelerator.prepare(
    #     args.model, args.optimizer, args.dataloaders["train"], args.dataloaders["val"], args.dataloaders["val-sqa3d"], args.dataloaders["val-scan2cap"], args.dataloaders["finetune"]
    # )
    args.model, args.optimizer = accelerator.prepare(args.model, args.optimizer)
    for k, v in args.dataloaders.items():
        if v is not None:
            args.dataloaders[k] = accelerator.prepare(v)

    # convert ME's batchnorm to sync batchnorm
    if args.pc_tokenizer_type == "minkowski":
        args.model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(args.model)
    # args.model.module.load_mnet()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(args.datasets["train"])), 5):
        logger.info(f"Sample {index} of the training set: {args.datasets['train'][index]}.")

    # Figure out how many steps we should save the Accelerator states
    if args.checkpointing_steps < 1: # which means it's a percentage of (one epoch)
        args.checkpointing_steps = args.total_steps * args.checkpointing_steps // args.epochs
    args.checkpointing_steps = int(args.checkpointing_steps)

    logger.info(f"Checkpointing every {args.checkpointing_steps} steps.")
    
    checkpointing_steps = args.checkpointing_steps

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(args.datasets['train'])}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.total_steps}")
    logger.info(f"  Checkpointing every {checkpointing_steps} steps")


    if accelerator.is_local_main_process:
        wandb.init(project=wandb_project, name=run_name, config=args, config_exclude_keys=[
            "accelerator", "model", "processor", 
            "optimizer", "scheduler", "datasets", "dataloaders",
            # "constraint",
            ])

    # avoid record to wandb
    if args.restrict_vocab:
        tokenizer: transformers.LlamaTokenizerFast = processor.tokenizer
        args.generation_config["constraints"] = [transformers.DisjunctiveConstraint([
            tokenizer.encode(f"{v}|ENDOFTEXT|", add_special_tokens=False) for v in args.answer_vocab
        ])]
        # extremely slow.

    if args.validate_at_start:
        validate(args, 0)

    for epoch in range(args.epochs):
        args.is_finetuning = epoch >= (args.epochs - args.finetune_epochs)
        train(args, epoch)
        # validate(args, epoch)

        # save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        unwrapped_model: Fuyu3DCausalLMv2 = accelerator.unwrap_model(args.model)
        unwrapped_model.save_pretrained(output_dir)
    # validate(args, args.epochs - 1) # validate at the end of training

    if accelerator.is_local_main_process:
        wandb.finish()