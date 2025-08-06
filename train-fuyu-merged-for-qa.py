import transformers
from transformers import FuyuProcessor, FuyuForCausalLM, AutoModelForCausalLM, FuyuConfig, AutoTokenizer, BertTokenizer, LlamaTokenizerFast
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
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from typing import List, Set, Tuple, Dict, Union, Any, Optional
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from copy import deepcopy
from utils.pc_utils import random_sampling, rotx, roty, rotz
import pickle
try:
    import MinkowskiEngine as ME
except ImportError:
    print("MinkowskiEngine is not installed.")
    ME = None
from icecream import ic
from models.fuyu_3d import Fuyu3DCausalLMv2, MyObjectDict, average_meter
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
    ScanNetFrameQADataset,
    clean_answer,
    print_once,
    gather_scalar,
    SVC_PATH,
)


MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
LABEL_START_TOKEN = ""


# --- setup wandb ---
wandb_project = "Kuri3D-merged-qa"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

base_model_name = "fuyu-8b"
project = "scanqa"
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = base_model_name + "-" + project + "-" + datetime_str
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
output_name = f"{run_name}-{datetime_str}"
output_dir = os.path.join(CURRENT_DIR, "..", "kuri3d-output", output_name)
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

def parse_args():
    parser = ArgumentParser()
    # Data
    parser.add_argument("--i2t_scanqa", type=str, default=f"{SVC_PATH}/i2t/scene_eval_decl_gpt3.5_aligned_scanqa_qonly_all_video_2.json")
    parser.add_argument("--i2t_sqa3d", type=str, default=f"{SVC_PATH}/i2t/scene_eval_sqa_video_qonly.pkl")
    parser.add_argument("--i2t_scan2cap", type=str, default=f"{SVC_PATH}/i2t/scene_bbox_view_map_full.json")
    parser.add_argument("--i2t_scan2cap_val", type=str, default=f"{SVC_PATH}/i2t/scene_bbox_view_map_for_valtest_mask3d.json")
    parser.add_argument("--i2t_scanrefer", type=str, default=f"{SVC_PATH}/i2t/scene_best_view_for_grounding.pth")
    parser.add_argument("--i2t_scanqa_mv", type=str, default=f"{SVC_PATH}/i2t/scene_eval_scanqa_mv.pth")
    

    parser.add_argument("--frame_path_scanqa", type=str, default=f"{SVC_PATH}/frames_square")
    parser.add_argument("--frame_path_sqa3d", type=str, default=f"{SVC_PATH}/frames_square")
    parser.add_argument("--frame_path_scan2cap", type=str, default=f"{SVC_PATH}/frames_square")

    
    parser.add_argument("--use_augment", action="store_true")
    parser.add_argument("--finetune_epochs", type=int, default=0)
    
    parser.add_argument("--add_scanqa", action="store_true")
    parser.add_argument("--add_scan2obj", action="store_true")
    parser.add_argument("--add_nr3d", action="store_true")
    parser.add_argument("--add_sr3d", action="store_true")
    parser.add_argument("--add_scan2cap", action="store_true")
    parser.add_argument("--add_sqa3d", action="store_true")
    parser.add_argument("--add_lamm3d", action="store_true")
    parser.add_argument("--add_scenecap", action="store_true")
    parser.add_argument("--add_framecap", action="store_true")
    parser.add_argument("--add_frameqa", action="store_true")
    parser.add_argument("--framecap_percentile", type=str, default="30.0")
    parser.add_argument("--framecap_name", type=str, default="framecap")
    parser.add_argument("--scan2cap_predicted_bbox_file", type=str, default=f"{SVC_PATH}/i2t/scene_bbox_info_for_valtest_mask3d.pkl")
    parser.add_argument("--add_nr3d_val", action="store_true")
    parser.add_argument("--add_nr3d_val_for_training", action="store_true")
    parser.add_argument("--add_sr3d_val_for_training", action="store_true")
    parser.add_argument("--add_framecap_val_for_training", action="store_true")

    parser.add_argument("--add_scanrefer", action="store_true")
    parser.add_argument("--add_scanrefer_nr3d", action="store_true")
    parser.add_argument("--add_scanrefer_sr3d", action="store_true")
    parser.add_argument("--add_scanrefer_train_for_val", action="store_true")

    parser.add_argument("--add_scanqa_test", action="store_true")
    parser.add_argument("--add_scan2obj_val", action="store_true")
    parser.add_argument("--add_scanqa_mv", action="store_true")

    parser.add_argument("--scan2cap_metric_type", type=str, default="recall")
    parser.add_argument("--deduplicate_captions", action="store_true")
    parser.add_argument("--merged_scan2obj", action="store_true")

    parser.add_argument("--use_annealing_data_schedule", action="store_true")

    parser.add_argument("--clean_qa_answer", action="store_true")

    # LLM or LVLM
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--use_phi3v", action="store_true")

    
    # Modality/Prompt
    parser.add_argument("--sample_with_sqrt_freq", action="store_true")
    parser.add_argument("--use_no_location_text", action="store_true")
    parser.add_argument("--label_start_token", type=str, default="\x04")
    parser.add_argument("--prompt_end_token", type=str, default="")
    parser.add_argument("--use_no_dataset_name", action="store_true")
    parser.add_argument("--framecap_as_input", action="store_true")
    parser.add_argument("--scale_bbox", type=int, default=100)
    parser.add_argument("--use_object_index_embedding", action="store_true")
    parser.add_argument("--shuffle_objects", action="store_true")
    parser.add_argument("--use_separated_grounding_indices", action="store_true") # TODO
    parser.add_argument("--keep_bad_start_words", action="store_true")
    parser.add_argument("--use_object_textual_index", action="store_true")
    parser.add_argument("--added_object_tokens", type=int, default=-1)
    parser.add_argument("--multiple_input_images", type=str, default="2x2", help="HxW, the layout of multiple input images")
    parser.add_argument("--mv_for_scanqa", action="store_true", help="Use multiview for ScanQA")
    parser.add_argument("--use_birdview", action="store_true", help="Use birdview rather than paired images")
    parser.add_argument("--birdview_path", type=str, default=f"{SVC_PATH}/bird_views_cropped")
    parser.add_argument("--use_object_index_for_scan2cap", action="store_true", help="Use object index in prompt for Scan2Cap")


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
    parser.add_argument("--remove_token_range", type=str, default="")

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
    parser.add_argument("--use_related_object_bbox", action="store_true")
    parser.add_argument("--choose_related_objects", action="store_true")
    parser.add_argument("--not_trim_objects", action="store_true")
    parser.add_argument("--iosa_threshold", type=float, default=0.15)

    parser.add_argument("--not_use_2d", action="store_true")
    parser.add_argument("--not_use_3d", action="store_true")

    parser.add_argument("--p_drop_2d", type=float, default=0.0)
    parser.add_argument("--p_drop_3d", type=float, default=0.0)
    parser.add_argument("--do_drop_2d_partial", action="store_true")
    parser.add_argument("--p_drop_2d_partial_alpha", type=float, default=2.0)
    parser.add_argument("--p_drop_2d_partial_beta", type=float, default=8.0)

    parser.add_argument("--keep_all_objects", action="store_true")

    parser.add_argument("--use_grounding_classifier", action="store_true")
    parser.add_argument("--coeff_grounding_classifier", type=float, default=0.5)
    
    #  |- Qformer
    parser.add_argument("--num_query_tokens", type=int, default=128)
    parser.add_argument("--qformer_num_hidden_layers", type=int, default=12)
    parser.add_argument("--pretrained_qformer", type=str, default=f"{SVC_PATH}/bert-base-embedding")
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
    parser.add_argument("--lora_word_embedding", action="store_true")

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

    # Finetune
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--trainable_lora_in_finetune", action="store_true")
    parser.add_argument("--create_new_lora_for_finetune", action="store_true")
    parser.add_argument("--validate_at_start", action="store_true")
    parser.add_argument("--special_finetune_prompt", type=str, default="")
    parser.add_argument("--only_load_adapter", action="store_true")

    parser.add_argument("--no_save", action="store_true", help="Do not save the model, useful in full-size finetuning, and the model is too large to save.")
    
    logger.info(f"Output directory: {output_dir}")

    return parser.parse_args()

def get_model(args):
    # load model. NOTE: in bfloat16
    in_channels = 128 * int(args.use_multiview) + int(args.use_height) + 3 * int(args.use_normal) + 3 * int(args.use_color)
    if args.pc_tokenizer_type == "minkowski":
        in_channels += 3 # concat xyz

    logger.info(f"Using {in_channels} channels for 3D data.")

    model_id = f"{SVC_PATH}/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)

    if not args.detector_from_scratch:
        model.load_detector()

    if args.checkpoint_path != "":
        logger.info(f"Loading checkpoint from {args.checkpoint_path}...")
        model.load_pretrained(args.checkpoint_path)

    print(model)
    return model, processor 

def get_peft_fuyu(model, args):
    if args.lora_rank == 0:
        # freeze all parameters
        logger.info(f"No LoRA applied as lora_rank == {args.lora_rank}.")
        logger.info("Freezing all LVLM parameters...")
        for p in model.fuyu.parameters():
            p.requires_grad = False
        return model

    elif args.lora_rank == -1:
        # full fine-tuning
        logger.info("Full fine-tuning.")
        for p in model.fuyu.parameters():
            p.requires_grad = True
        return model

    modules_to_save = [
        "mnet", 
        "linear_3d", "linear_focus_bbox", "think_tokens", 
        "object_index_embedding", "related_object_mlp", "related_object_embedding",  
        "frame_params_head",
    ]
    if args.unfreeze_word_embedding:
        if args.lora_word_embedding:
            args.lora_target_modules = "embed_tokens,lm_head," + args.lora_target_modules
        else:
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
        target_modules=args.lora_target_modules.split(","),
        modules_to_save=modules_to_save,
        init_lora_weights="pissa_niter_48" if args.use_pissa else True,
    )

    # apply LORA to the LLM
    if getattr(model, "fuyu", None) is not None:
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
                original_device = model.fuyu.device
                model.fuyu = model.fuyu.cuda() # merge on GPU for speed
                model.fuyu = model.fuyu.merge_and_unload(progressbar=True)
                model.fuyu = model.fuyu.to(original_device)
                
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
    generated = model.generate( **model_inputs, **generation_config, max_new_tokens=max_new_tokens, synced_gpus=False)[:, -max_new_tokens:]

    model_outputs: List[str] = processor.tokenizer.batch_decode(generated, skip_special_tokens=skip_special_tokens)
    if any(['\x04' in m for m in model_outputs]):
        prediction = [m.split('\x04', 1)[1].strip() if '\x04' in m else m for m in model_outputs]
    elif any(['[/INST]' in m for m in model_outputs]):
        prediction = [m.split('[/INST]', 1)[1].strip() if '[/INST]' in m else m for m in model_outputs]
    elif any(["<|assistant|>" in m for m in model_outputs]):
        prediction = [m.split('<|assistant|>', 1)[1].strip() if '<|assistant|>' in m else m for m in model_outputs]
    else:
        prediction = model_outputs

    if return_text:
        return prediction
    else:
        return {
            "model_outputs": generated,
            "prediction": prediction
        }


def get_image_for_question(frame_path, scene_to_image, whole_qid):
    scene_id = whole_qid.split("-")[1] # train-xxxx-xxxx
    image = scene_to_image[whole_qid][0] # xxxx.jpg
    image_path = os.path.join(frame_path, f"{scene_id}_00/color/{image}")
    return Image.open(image_path)

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
        "shuffle_objects": args.shuffle_objects,
        "use_birdview": args.use_birdview,
        "birdview_path": args.birdview_path,
        # "frozen_object_feature_path": args.frozen_object_feature_path if args.use_frozen_object_feature else None,
    }

    shared_densecap_config = {
        "use_no_location_text": args.use_no_location_text,
        "use_no_dataset_name": args.use_no_dataset_name,
        "remove_bad_start_words": not args.keep_bad_start_words,
        "use_object_index": args.use_object_index_for_scan2cap,
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
            multiple_input_images=args.multiple_input_images if args.mv_for_scanqa else "1x1",
            **shared_config,
        )
        datasets.append(scanqa_train)

    if args.add_scanqa_mv:
        scanqa_mv_train = ScanQADatasetUnified(
            name="scanqa-mv",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanqa_mv,
            views_path=args.frame_path_scanqa,
            multiple_input_images=args.multiple_input_images,
            **shared_config,
        )
        datasets.append(scanqa_mv_train)

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
            # use_no_location_text=args.use_no_location_text,
            **shared_config,
            **shared_densecap_config
        )
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

    if args.add_nr3d_val_for_training:
        scan2cap_nr3d_val_dataset = Scan2CapSimpleDataset(
            name="scan2cap-nr3d",
            split="val",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            **shared_densecap_config,
            **shared_config,
        )
        if args.deduplicate_captions:
            scan2cap_nr3d_val_dataset.deduplicate_captions()
        datasets.append(scan2cap_nr3d_val_dataset)

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


    if args.add_sr3d_val_for_training:
        scan2cap_sr3d_val_dataset = Scan2CapSimpleDataset(
            name="scan2cap-sr3d",
            split="val",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap,
            views_path=args.frame_path_scan2cap,
            # use_no_location_text=args.use_no_location_text,
            **shared_densecap_config,
            **shared_config,
        )
        if args.deduplicate_captions:
            scan2cap_sr3d_val_dataset.deduplicate_captions()
        datasets.append(scan2cap_sr3d_val_dataset)

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

    if args.add_frameqa:
        frameqa_dataset = ScanNetFrameQADataset(
            name="frameqa",
            dataset_type="frameqa",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scan2cap, # dummy
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
        datasets.append(frameqa_dataset)

    if args.add_framecap_val_for_training:
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
        datasets.append(val_framecap_dataset)

    if args.add_scanrefer:
        scanrefer_dataset = ScanReferDataset(
            # name="scanrefer-resplit-small",
            name="scanrefer",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            # use_object_index=args.use_object_index_for_scan2cap,
            **shared_config,
        )
        datasets.append(scanrefer_dataset)

    if args.add_scanrefer_nr3d:
        scanrefer_nr3d_dataset = ScanReferDataset(
            name="scanrefer-nr3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
        datasets.append(scanrefer_nr3d_dataset)

    if args.add_scanrefer_sr3d:
        scanrefer_sr3d_dataset = ScanReferDataset(
            name="scanrefer-sr3d",
            split="train",
            use_augment=args.use_augment,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
        datasets.append(scanrefer_sr3d_dataset)
    
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
            multiple_input_images=args.multiple_input_images if args.mv_for_scanqa else "1x1",
            **shared_config,
        )
    else:
        val_dataset = None

    if args.add_scanqa_mv:
        val_dataset_mv = ScanQADatasetUnified(
            name="scanqa-mv",
            split="val",
            use_augment=False,
            i2t=args.i2t_scanqa_mv,
            views_path=args.frame_path_scanqa,
            multiple_input_images=args.multiple_input_images,
            **shared_config,
        )
    else:
        val_dataset_mv = None

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
        val_dataset_scan2cap_gt = None

    else:
        val_dataset_scan2cap = None
        val_dataset_scan2cap_gt = None

    if args.add_scanqa_test:
        test_scanqa_w_obj = ScanQADatasetUnified(
            name="scanqa",
            split="test_w_obj",
            use_augment=False,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            multiple_input_images=args.multiple_input_images if args.mv_for_scanqa else "1x1",
            **shared_config,
        )

        test_scanqa_wo_obj = ScanQADatasetUnified(
            name="scanqa",
            split="test_wo_obj",
            use_augment=False,
            i2t=args.i2t_scanqa,
            views_path=args.frame_path_scanqa,
            multiple_input_images=args.multiple_input_images if args.mv_for_scanqa else "1x1",
            **shared_config,
        )
    else:
        test_scanqa_w_obj = None
        test_scanqa_wo_obj = None

    if args.add_scan2obj_val:
        logger.warning("Scan2Obj validation dataset is not implemented yet.")

    if args.add_scanrefer_train_for_val:
        # NOTE: DEBUG
        shared_config_for_debug = deepcopy(shared_config)
        shared_config_for_debug["ratio"] = 0.01
        val_dataset_scanrefer_train = ScanReferDataset(
            name="scanrefer",
            split="train",
            # name="scanrefer-resplit-small",
            # split="train",
            use_augment=False,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            start_from_last=True,
            enforce_validation=True,
            **shared_config_for_debug,
            # **shared_config,
        )
    else:
        val_dataset_scanrefer_train = None

    if args.add_scanrefer:
        # NOTE: DEBUG, 0.1x ratio
        val_dataset_scanrefer = ScanReferDataset(
            # name="scanrefer-resplit-small",
            name="scanrefer",
            split="val",
            use_augment=False,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            # use_object_index=args.use_object_index_for_scan2cap,
            **shared_config,
        )
    else:
        val_dataset_scanrefer = None
    

    if (debug := False) and args.add_scanrefer_nr3d:
        val_dataset_scanrefer_nr3d = ScanReferDataset(
            name="scanrefer-nr3d",
            split="val",
            use_augment=False,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
    else:
        val_dataset_scanrefer_nr3d = None

    if (debug := False) and args.add_scanrefer_sr3d:
        val_dataset_scanrefer_sr3d = ScanReferDataset(
            name="scanrefer-sr3d",
            split="val",
            use_augment=False,
            i2t=args.i2t_scanrefer,
            views_path=args.frame_path_scan2cap,
            **shared_config,
        )
    else:
        val_dataset_scanrefer_sr3d = None
    
    if args.use_dummy_image_for_scan2cap:
        val_dataset_scan2cap.image_getter = dummy_image_getter
        val_dataset_scan2cap_gt.image_getter = dummy_image_getter
        val_dataset_nr3d.image_getter = dummy_image_getter


    # answer vocab
    if args.add_scanqa:
        answer_vocab = get_answer_vocab([scanqa_train, val_dataset]) # for scanqa eval
        args.answer_vocab = answer_vocab
    
    dataset_dict = {
        # "finetune": scanqa_train,
        "finetune": datasets[0], # FIXME: make it any dataset
        "train": train_dataset,
        "val": val_dataset,
        "val-scanqa-mv": val_dataset_mv,
        "val-sqa3d": val_dataset_sqa3d,
        "val-scan2cap": val_dataset_scan2cap,
        "test-sqa3d": test_dataset_sqa3d,
        "val-scan2cap-gt": val_dataset_scan2cap_gt,
        "val-nr3d": val_dataset_nr3d,
        "test-scanqa-w-obj": test_scanqa_w_obj,
        "test-scanqa-wo-obj": test_scanqa_wo_obj,
        "val-scanrefer": val_dataset_scanrefer,
        "val-scanrefer-nr3d": val_dataset_scanrefer_nr3d,
        "val-scanrefer-sr3d": val_dataset_scanrefer_sr3d,
        "val-scanrefer-train": val_dataset_scanrefer_train,
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
            if isinstance(examples[0]["object_feature"], np.ndarray):
                object_feature = torch.from_numpy(np.stack([e["object_feature"] for e in examples], axis=0)).float()
            elif isinstance(examples[0]["object_feature"], torch.Tensor):
                object_feature = torch.stack([e["object_feature"] for e in examples], dim=0)
            else:
                raise NotImplementedError("Unknown 3D data format with keys: " + str(examples[0].keys()))
            
            if isinstance(examples[0]["object_mask"], np.ndarray):
                object_mask = torch.from_numpy(np.stack([e["object_mask"] for e in examples], axis=0)).bool()
            elif isinstance(examples[0]["object_mask"], torch.Tensor):
                object_mask = torch.stack([e["object_mask"] for e in examples], dim=0).bool()
            else:
                raise NotImplementedError("Unknown 3D data format with keys: " + str(examples[0].keys()))
            # object_mask = torch.from_numpy(np.stack([e["object_mask"] for e in examples], axis=0)).bool()
            data = [object_feature, object_mask]
            if "predicted_bbox_corners" in examples[0] and examples[0]["predicted_bbox_corners"] is not None:
                predicted_bbox_corners = torch.from_numpy(np.stack([e["predicted_bbox_corners"] for e in examples], axis=0)).float()
                data.append(predicted_bbox_corners)

            return data
    
    raise NotImplementedError("Unknown 3D data format with keys: " + str(examples[0].keys()))

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def collate_vl(examples, is_eval=False) -> dict:
    global processor, qtokenizer
    texts = [e['target'] for e in examples]
    # print(texts)
    images = [e["image"] for e in examples]

    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    output = {k: v for k, v in output.items()}

    texts_instructions = [e['target_instruction'] for e in examples] # without \x04 and later tokens (gt annotations)
    qformer_inputs = qtokenizer(texts_instructions, padding="longest", return_tensors="pt")
    output["qformer_inputs"] = qformer_inputs

    output["target_object_indices"] = torch.tensor([e["object_index"] if "object_index" in e else 0 for e in examples])

    return output

def collate_frame_info(output, examples) -> dict:
    output["frame_intrinsics"] = torch.tensor(np.stack([e["frame_intrinsics"] for e in examples])).to(torch.float32)
    output["frame_poses"] = torch.tensor(np.stack([e["frame_poses"] for e in examples])).to(torch.float32)
    output["axis_alignments"] = torch.tensor(np.stack([e["axis_alignments"] for e in examples])).to(torch.float32)
    output["frame_caption_mask"] = torch.tensor(np.stack([e["frame_caption_mask"] for e in examples])).to(torch.bool)
    if GLOBAL_CONF.no_bbox_mask_for_framecap:
        output["frame_caption_mask"] = torch.zeros_like(output["frame_caption_mask"]) # mark as no sample is from frame caption
    
    if GLOBAL_CONF.use_related_object_bbox:
        output["related_object_bboxes"] = [
            torch.tensor(e["related_object_bboxes"]).to(torch.float32) if len(e) > 0 else None
            for e in examples
        ]
        

    return output

def collate_fn(examples) -> dict:
    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    output["focus_bbox"] = torch.tensor(np.stack([e["target_bbox"] for e in examples]))
    has_focus_bbox_types = ["scan2cap", "scan2obj", "scan2cap-nr3d", "scan2cap-sr3d"]
    has_focus_bbox_masks = [e["data_type"] in has_focus_bbox_types for e in examples]
    output["focus_bbox_mask"] = torch.tensor(has_focus_bbox_masks)

    positions = [] # label start position (before \x04)

    if LABEL_START_TOKEN == "\x04":
        start_token = GLOBAL_CONF.processor.tokenizer("\x04", add_special_tokens=False)["input_ids"][-1]
    elif LABEL_START_TOKEN == "[/INST]":
        start_token = GLOBAL_CONF.processor.tokenizer("[/INST]", add_special_tokens=False)["input_ids"][-2] 
    elif LABEL_START_TOKEN == "<|assistant|>\n":
        start_token = GLOBAL_CONF.processor.tokenizer("<|assistant|>\n", add_special_tokens=False)["input_ids"][-1]

    for input_ids in output["input_ids"]:
        if LABEL_START_TOKEN == "\x04":
            position = (input_ids == start_token).nonzero(as_tuple=True)[0][0]
        elif LABEL_START_TOKEN == "[/INST]":
            position = (input_ids == start_token).nonzero(as_tuple=True)[0][-1] + 1
            # print(position)
        elif LABEL_START_TOKEN == "<|assistant|>\n":
            position = (input_ids == start_token).nonzero(as_tuple=True)[0][-1]

        positions.append(position - 1) # -1 to consider that \x04? or +2 to not consider \x04 and the space after it
            # and for "INST" in "[/INST]", the label starts after the "]"

    output["labels"] = torch.full_like(output["input_ids"], -100)  # This creates a tensor filled with -100
    for i in range(len(positions)):
        output["labels"][i, positions[i]:] = output["input_ids"][i, positions[i]:] # mask the tokens before <s> in labels

    output["mnet_inputs"] = collate_3d(examples)

    return output

def collate_fn_eval(examples) -> dict:
    to_copy_keys = ["raw_question_id", "answers", "target_id", "scanrefer_id", "object_index"]

    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    to_copy_keys = list(set(to_copy_keys) & set(examples[0].keys())) # filter out keys that are not in examples
    for k in to_copy_keys:
        # output[k] = [e[k] for e in examples]
        output[k] = [e.get(k, "MISSING") for e in examples]
        
    output["to_copy_keys"] = to_copy_keys
    output["mnet_inputs"] = collate_3d(examples)
    
    return output

def collate_fn_eval_scan2cap(examples) -> dict:
    to_copy_keys = ["raw_question_id", "question_id", "target", "scan2cap_id", "description", "target_id"]

    output = collate_vl(examples)
    output = collate_frame_info(output, examples)

    output["focus_bbox"] = torch.tensor(np.stack([e["target_bbox"] for e in examples]))

    to_copy_keys = list(set(to_copy_keys) & set(examples[0].keys())) # filter out keys that are not in examples
    for k in to_copy_keys:
        # output[k] = [e[k] for e in examples]
        output[k] = [e.get(k, "MISSING") for e in examples]

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
            collate_fn_for_dataset = collate_fn_eval if split in [
                "val", "val-sqa3d", "test-sqa3d", "test-scanqa-w-obj", "test-scanqa-wo-obj",
                "val-scanrefer", "val-scanrefer-nr3d", "val-scanrefer-sr3d", "val-scanrefer-train", "val-scanqa-mv",
            ] else collate_fn_eval_scan2cap

        return DataLoader(
            dataset, 
            batch_size=args.batch_size if is_train_dataset else args.eval_batch_size,
            shuffle=is_train_dataset and not args.use_annealing_data_schedule, # if use anneal, shuffle is done in dataset
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
    name_to_params_group_map["params_adapter"] = ["linear_3d", "linear_focus_bbox", "think_tokens", "qformer", "qformer_query_tokens", "qformer_to_language"]
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
    return optimizer, scheduler
    

def train(args, epoch):
    logger.info(f"[{epoch}]start training...")
    args.model.train()
    torch.cuda.empty_cache()
    average_meter.reset()

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

                loss_grounding = getattr(outputs, "loss_grounding", None)
                grounding_acc = getattr(outputs, "grounding_acc", None)
                grounding_acc_positive = getattr(outputs, "grounding_acc_positive", None)
                grounding_acc_negative = getattr(outputs, "grounding_acc_negative", None)
                grounding_acc_softmax = getattr(outputs, "grounding_acc_softmax", None)
                if loss_grounding is not None:
                    loss_grounding = accelerator.gather(loss_grounding.unsqueeze(0)).mean().item()
                    grounding_acc = gather_scalar(accelerator, grounding_acc)
                    grounding_acc_positive = gather_scalar(accelerator, grounding_acc_positive)
                    grounding_acc_negative = gather_scalar(accelerator, grounding_acc_negative)
                    grounding_acc_softmax = gather_scalar(accelerator, grounding_acc_softmax)
                else:
                    loss_grounding = 0
                    grounding_acc = 0
                    grounding_acc_positive = 0
                    grounding_acc_negative = 0
                    grounding_acc_softmax = 0

                if accelerator.is_local_main_process:
                    wandb.log({
                        "train/loss": args.history_losses[-1], 
                        "train/loss_frame": loss_frame,
                        "train/loss_grounding": loss_grounding,
                        "train/global_step": len(args.history_losses),
                        "train/seen_samples": len(args.history_losses) * args.batch_size * args.accelerator.num_processes,
                        "train/lr": args.scheduler.get_last_lr()[0],
                        "train/grounding_acc": grounding_acc,
                        "train/grounding_acc_positive": grounding_acc_positive,
                        "train/grounding_acc_negative": grounding_acc_negative,
                        "train/grounding_acc_softmax": grounding_acc_softmax,
                    })

            if args.gradient_clipping > 1e-8:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            args.optimizer.step()
            args.scheduler.step()
            args.optimizer.zero_grad()

        if step % args.print_log_step == 0:
            if accelerator.is_local_main_process:
                print(f"[{epoch}/{len(args.history_losses)}-{args.total_steps}]loss: {args.history_losses[-1]}, lr: {args.scheduler.get_last_lr()[0]}, loss_frame: {loss_frame}, loss_grounding: {loss_grounding}, acc: {grounding_acc:.6f}, acc_pos: {grounding_acc_positive:.6f}, acc_neg: {grounding_acc_negative:.6f}, acc_softmax: {grounding_acc_softmax:.6f}")

        if len(args.history_losses) % args.checkpointing_steps == 0: # validate at the first step
            # validate
            logger.info(f"[{epoch}]validating @ step {len(args.history_losses)}...")

            # save checkpoint
            # only save PEFT lora
            # only save when full training
            if args.train_ratio == 1 and not args.no_save:
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
    average_meter.reset()
    accelerator: Accelerator = args.accelerator

    
    all_metrics = {}
    for kind, dataloader in {
        "scanqa-test-w-obj": args.dataloaders["test-scanqa-w-obj"],
        "scanqa-test-wo-obj": args.dataloaders["test-scanqa-wo-obj"],
        "scan2cap": args.dataloaders["val-scan2cap"],
        "scanqa": args.dataloaders["val"],
        "scanqa-mv": args.dataloaders["val-scanqa-mv"],
        "sqa3d": args.dataloaders["val-sqa3d"],
        "sqa3d-test": args.dataloaders["test-sqa3d"],
        "scan2cap-gt": args.dataloaders["val-scan2cap-gt"],
        "nr3d": args.dataloaders["val-nr3d"],
        "scanrefer": args.dataloaders["val-scanrefer"],
        "scanrefer-nr3d": args.dataloaders["val-scanrefer-nr3d"],
        "scanrefer-sr3d": args.dataloaders["val-scanrefer-sr3d"],
        "scanrefer-train": args.dataloaders["val-scanrefer-train"],
    }.items():
        if dataloader is None:
            continue

        preds, target_texts = {}, {}
        logger.info(f"[{epoch}]validating on {kind}...")
        for step, batch in enumerate(dataloader):

            with torch.no_grad():
                batch_size = batch["input_ids"].size(0)

                to_copy_keys = batch.pop("to_copy_keys")
                copied_dict = {k: deepcopy(batch[k]) for k in to_copy_keys}
                for k in to_copy_keys:
                    del batch[k]

                # if kind in ["scanqa", "sqa3d", "sqa3d-test", "scanqa-test-w-obj", "scanqa-test-wo-obj"]:
                if kind in ["scanqa", "sqa3d", "sqa3d-test", "scanqa-mv"]:
                    target_text = copied_dict["answers"]
                elif kind in ["scan2cap", "scan2cap-gt", "nr3d"]:
                    target_text = copied_dict["description"]
                elif kind in ["scanqa-test-w-obj", "scanqa-test-wo-obj"]:
                    target_text = [["dummy"]] * batch_size # scanqa test set does not have target text
                elif kind in ["scanrefer", "scanrefer-nr3d", "scanrefer-test", "scanrefer-train"]:
                    target_text = (copied_dict["target_id"], copied_dict["object_index"])
                else:
                    raise NotImplementedError(f"kind {kind} not implemented.")
                
                # inference by generate
                unwrapped_model = accelerator.unwrap_model(args.model, keep_fp32_wrapper=True)
                if accelerator.is_local_main_process:
                    transformers.utils.logging.set_verbosity_error() # disable logging
                pred_answer = batch_generate_v2(
                    unwrapped_model, 
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

                if kind in ["scanqa", "sqa3d", "sqa3d-test", "scan2cap-gt", "scanqa-test-w-obj", "scanqa-test-wo-obj", "scanqa-mv"]:
                    for i in range(len(pred_answer)):
                        # random_uid = str(uuid.uuid4())
                        raw_question_id = copied_dict["raw_question_id"][i]
                        if kind in ["scan2cap-gt"]:
                            # preds[random_uid] = [preprocess_sos_eos_for_scan2cap(pred_answer[i])] # for evaluation, it shall be a list of predictions
                            preds[raw_question_id] = [
                                postprocess_punctuation_for_caption_metrics(
                                    preprocess_sos_eos_for_scan2cap(pred_answer[i])
                                )
                            ]
                            target_texts[raw_question_id] = deepcopy(args.datasets["val-scan2cap-gt"].corpus[copied_dict["scan2cap_id"][i]])
                        else:
                            preds[raw_question_id] = [pred_answer[i]] # for evaluation, it shall be a list of predictions
                            target_texts[raw_question_id] = target_text[i]
                elif kind in ["scan2cap", "nr3d"]:
                    for pred, sample_id in zip(pred_answer, copied_dict["scan2cap_id"]):
                        # preds[sample_id] = [preprocess_sos_eos_for_scan2cap(pred)] # for evaluation, it shall be a list of predictions
                        preds[sample_id] = [
                            postprocess_punctuation_for_caption_metrics(
                                preprocess_sos_eos_for_scan2cap(pred)
                            )
                        ]
                elif kind in ["scanrefer", "scanrefer-nr3d", "scanrefer-sr3d", "scanrefer-train"]:
                    for i, (pred, sample_id) in enumerate(zip(pred_answer, copied_dict["scanrefer_id"])):
                        preds[sample_id] = pred # this shall be a prediction of bbox index (of input boxes)
                        target_texts[sample_id] = target_text[0][i]
                else:
                    raise NotImplementedError(f"kind {kind} not implemented.")
                
        # evaluate
        # gather all predictions and targets
        if accelerator.is_local_main_process:
            print("Gathering predictions and targets...")
        preds = gather_object([(k, v) for k, v in preds.items()])
        preds = {k: v for k, v in preds}
        metrics_epoch = {}
        metric_message = ""
        if kind in ["scanqa", "sqa3d", "sqa3d-test", "scan2cap-gt", "scanqa-test-w-obj", "scanqa-test-wo-obj", "scanqa-mv"]:
            # text similarity
            target_texts = gather_object([(k, v) for k, v in target_texts.items()])
            target_texts = {k: v for k, v in target_texts}
            # replace empty string with "empty"
            for k, v in target_texts.items():
                for i, vv in enumerate(v):
                    if len(vv.strip()) == 0:
                        target_texts[k][i] = "empty"

            if kind in ["scanqa"] and args.clean_qa_answer:
                # clean both gt and pred, this is done in LEO and ChatScene when evaluation on validation set
                # -- no big difference though
                print_once("Cleaning QA answers...")
                target_texts = {k: [clean_answer(vv) for vv in v] for k, v in target_texts.items()}
                preds = {k: [clean_answer(vv) for vv in v] for k, v in preds.items()}

            if kind in ["scanqa-mv"]:
                # split eval for different view difficulty
                dataset: ScanQADatasetUnified = dataloader.dataset
                question_id_to_difficulty = dataset.get_question_id_to_difficulty() # question_id -> difficulty
                difficulty_levels = set(question_id_to_difficulty.values())
                logger.info(f"Difficulty levels: {difficulty_levels}")
                # for difficulty in sorted(difficulty_levels): # 0, 1, 2, 3, ...
                # merge last two difficulties
                for i_diff, difficulty in enumerate(sorted(difficulty_levels)[:-1]):
                    question_ids = set([k for k, v in question_id_to_difficulty.items() if v == difficulty])
                    if i_diff == len(difficulty_levels) - 2:
                        question_ids.update(
                            [k for k, v in question_id_to_difficulty.items() if v == sorted(difficulty_levels)[-1]]
                        )
                    preds_difficulty = {k: preds[k] for k in question_ids}
                    target_texts_difficulty = {k: target_texts[k] for k in question_ids}
                    score_per_caption, metric_message_difficulty, metrics_epoch_difficulty = score_captions(target_texts_difficulty, preds_difficulty)
                    metrics_epoch_difficulty = {
                        f"{k}_N={difficulty}": v for k, v in metrics_epoch_difficulty.items()
                    }
                    metrics_epoch.update(metrics_epoch_difficulty)
                    metric_message += f"\nScoring QA with difficulty={difficulty}...\n"
                    metric_message += metric_message_difficulty

                    em, refined_em = compute_qa_score(preds_difficulty, target_texts_difficulty)
                    metrics_epoch[f"em_N={difficulty}"] = em
                    metrics_epoch[f"refined_em_N={difficulty}"] = refined_em

            score_per_caption, metric_message_new, metrics_epoch_new = score_captions(target_texts, preds)
            metrics_epoch.update(metrics_epoch_new)
            metric_message += metric_message_new

            # exact match
            em, refined_em = compute_qa_score(preds, target_texts)
            metrics_epoch["em"] = em
            metrics_epoch["refined_em"] = refined_em
        
        elif kind in ["scan2cap", "nr3d"]:
            # caption target is the corpus itself
            # score_per_caption, metric_message, metrics_epoch = score_captions(args.datasets["val-scan2cap"].corpus, preds)
            scan2cap_dataset: Scan2CapTestDataset = args.datasets["val-scan2cap"] if kind == "scan2cap" else args.datasets["val-nr3d"]
            # metrics_epoch = {}
            # metric_message = ""
            for iou in [0.25, 0.5]:
                gt_corpus, pred_corpus = scan2cap_dataset.process_predictions(preds, iou_threshold=iou, method=args.scan2cap_metric_type)
                # print(pred_corpus.keys(), gt_corpus.keys())
                score_per_caption, metric_message_iou, metrics_epoch_iou = score_captions(gt_corpus, pred_corpus)

                metric_message += f"\nScoring captions with iou={iou}...\n"
                metric_message += metric_message_iou
                metrics_epoch.update({
                    f"{k}@{iou}": v for k, v in metrics_epoch_iou.items()
                })
        elif kind in ["scanrefer", "scanrefer-nr3d", "scanrefer-sr3d", "scanrefer-train"]:
            scanrefer_dataset = dataloader.dataset

            target_texts = gather_object([(k, v) for k, v in target_texts.items()])
            target_texts = {k: v for k, v in target_texts} # target ids, of bbox array (not real object id)

            for iou in [0.25, 0.5]:
                # preds are bbox indices of input bbox
                # target_texts are GT bbox indices.
                metric_message_iou, metrics_epoch_iou = scanrefer_dataset.evaluate(preds, target_texts, iou_threshold=iou)

                metric_message += f"Scoring grounding with iou={iou}...\n"
                metric_message += metric_message_iou + "\n"
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
        if accelerator.is_local_main_process and not args.no_save:
            print(f"New best {args.best_criteria}: {args.best_metrics[args.best_criteria]}")
            # save checkpoint
            output_dir_ckpt = os.path.join(output_dir, f"best-{args.best_criteria}")
            if not os.path.exists(output_dir_ckpt):
                os.makedirs(output_dir_ckpt, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(args.model)
            unwrapped_model.save_pretrained(output_dir_ckpt)
            
    if accelerator.is_local_main_process:
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
    average_meter.reset()


if __name__ == "__main__":
    args = parse_args()
    GLOBAL_CONF = args

    device = args.device

    args.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    args.slurm_gpus = os.environ.get("SLURM_GPUS", None)
        
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    args.accelerator = accelerator
    average_meter.accelerator = accelerator
    
    state = accelerator.state
    if accelerator.is_local_main_process:
        print(args)

    # set log options, disable print for other processes
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
        _old_print = print
        print = lambda *args, **kwargs: None

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

    datasets = get_trainval_datasets(args)
    args.datasets = datasets
    dataloaders = get_trainval_dataloaders(args)
    args.dataloaders = dataloaders

    accelerator.wait_for_everyone()

    model, processor = get_model(args)
    # model = get_shrinked_model(model, args) # TODO: do this after accelerator.prepare, but LoRA needs to be done after shrink if lora word embeddings!!
    model = get_peft_fuyu(model, args)

    if len(args.prompt_end_token) == 0:
        args.prompt_end_token = processor.tokenizer.eos_token
    
    if args.prompt_end_token != processor.tokenizer.eos_token:
        logger.warning(f"Prompt end token: {args.prompt_end_token} instead of {processor.tokenizer.eos_token}")

    if args.use_llm:
        LABEL_START_TOKEN = "[/INST]"
    elif args.use_phi3v:
        LABEL_START_TOKEN = "<|assistant|>\n"
    else:
        LABEL_START_TOKEN = "\x04"

    logger.info(f"LLM/LVLM LABEL_START_TOKEN: {LABEL_START_TOKEN}")


    args.model = model
    args.processor = processor

    # set pad_token_id and eos_token_id
    args.generation_config["eos_token_id"] = processor.tokenizer.eos_token_id
    args.generation_config["pad_token_id"] = processor.tokenizer.pad_token_id
    
    # init instructblip tokenizer
    qtokenizer: BertTokenizer = BertTokenizer.from_pretrained(args.pretrained_qformer)
    args.qtokenizer = qtokenizer

    logger.info(f"Total {len(args.datasets['train'])} training samples.")
    logger.info(f"Total {len(args.datasets['finetune'])} finetune samples.")
    total_val_size = sum([len(v) for k, v in args.datasets.items() if (k.startswith("val") or k.startswith("test")) and v is not None])
    logger.info(f"Total {total_val_size} validation samples.")
    # calc total steps
    total_pretrain_steps = (args.epochs - args.finetune_epochs) * len(args.datasets["train"]) // args.batch_size // accelerator.num_processes
    total_finetune_steps = args.finetune_epochs * len(args.datasets["finetune"]) // args.batch_size // accelerator.num_processes
    args.total_steps = total_pretrain_steps + total_finetune_steps
    args.warmup_steps = min(args.total_steps // args.epochs // 5 if args.epochs > 0 else 1, 500)
    logger.info(f"Total {total_pretrain_steps} pretrain steps.")
    logger.info(f"Total {total_finetune_steps} finetune steps.")
    logger.info(f"Total {args.total_steps} training steps.")
    logger.info(f"Total {args.warmup_steps} warmup steps.")

    args.history_losses = []
    args.best_metrics = {}

    optimizer, scheduler = get_optimizer_scheduler(args, model)
    args.optimizer = optimizer
    args.scheduler = scheduler

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
    if args.checkpointing_steps <= 1: # which means it's a percentage of (one epoch)
        if args.epochs > 0:
            args.checkpointing_steps = args.total_steps * args.checkpointing_steps // args.epochs
        else:
            args.checkpointing_steps = 1
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


    if args.validate_at_start:
        validate(args, 0)

    for epoch in range(args.epochs):
        args.is_finetuning = epoch >= (args.epochs - args.finetune_epochs)
        train(args, epoch)

        # save
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not args.no_save:
            unwrapped_model: Fuyu3DCausalLMv2 = accelerator.unwrap_model(args.model)
            unwrapped_model.save_pretrained(output_dir)


    # validate(args, args.epochs - 1) # validate at the end of training

    if accelerator.is_local_main_process:
        wandb.finish()
