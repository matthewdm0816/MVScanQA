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
import pretty_errors
from fuyu_utils import (
    get_optimizer_param_groups_by_names_dict, 
    ScanQASQA3DDataset, 
    random_sampling, 
    rotx, 
    roty, 
    rotz, 
    VisualInstructionTuningDataset3D, 
    ScanReferDataset,
    get_3d_box,
    acc_iou,
    batch_parse_grounding_text,
    batch_parse_get_iou,
    calculate_reinforce_reward_labels,
    batch_calculate_reinforce_reward_labels,
)
from easydict import EasyDict as edict

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

wandb_project = "Kuri3D-grounding"
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
GLOBAL = edict()

def parse_args():
    parser = ArgumentParser()
    # Data
    # parser.add_argument("--prompt", default="Answer the following VQAv2 question based on the image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    parser.add_argument("--i2t", type=str, default="/scratch/mowentao/BLIP/scene_eval_scanrefer.pkl")
    parser.add_argument("--frame_path", type=str, default="/scratch/generalvision/ScanQA-feature/frames_square/")
    parser.add_argument("--dataset", type=str, default="scanrefer")
    # parser.add_argument("--sqa_prompt", type=str, default="Answer the following SQA3D question based on the situation and image:{}\x04 {}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
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
    parser.add_argument("--best_criteria", type=str, default="acc@0.25")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--reinforce_sigma", type=float, default=0.5, help="reward decay sigma")

    # Logging
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--print_log_step", type=int, default=50)


    # 3D options
    parser.add_argument("--use_3d", action="store_true")
    parser.add_argument("--spatial_patch_size", type=int, default=24)
    parser.add_argument("--voxel_size", type=float, default=0.02)
    parser.add_argument("--pooling_method", type=str, default="max")
    parser.add_argument("--shift_bbox_to_positive", action="store_true")

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

    if args.use_3d:

        model = Fuyu3DCausalLMv2(
            pretrained_args={
                "pretrained_model_name_or_path": model_id,
                "torch_dtype": torch.bfloat16,
            },
            mnet_path="/scratch/generalvision/mowentao/ScanQA/weights.pth",
            freeze_mnet=args.lr_3d <= 1e-8,
            spatial_patch_size=args.spatial_patch_size,
            pooling_method=args.pooling_method,
            vocab=processor.tokenizer.get_vocab(),
            reinforce=args.reinforce,
            reinforce_sigma=args.reinforce_sigma,
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


def batch_generate_v2(model: FuyuForCausalLM, model_inputs, max_new_tokens=80, return_text=True, skip_special_tokens=False, generation_config={}):
    # model_inputs = model_inputs.to('cuda')

    generated = model.generate( **model_inputs, **generation_config, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)[:, -max_new_tokens:]

    # print(processor.batch_decode(generated, skip_special_tokens=False))
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

def get_trainval_datasets(args) -> dict[str, Dataset]:
    if args.dataset == "scanrefer":
        assert args.use_3d, "scanqa_sqa only works with 3D"
        train_dataset = ScanReferDataset(
            name=args.dataset,
            split="train",
            ratio=args.train_ratio,
            use_color=True,
            use_augment=args.use_augment,
            use_multiview=False,
            use_height=False,
            use_normal=False,
            i2t=args.i2t,
            views_path=args.frame_path,
            shift_bbox_to_positive=args.shift_bbox_to_positive,
        )
        val_dataset = ScanReferDataset(
            name=args.dataset,
            split="val",
            ratio=args.train_ratio,
            use_color=True,
            use_augment=False,
            use_multiview=False,
            use_height=False,
            use_normal=False,
            i2t=args.i2t,
            views_path=args.frame_path,
            shift_bbox_to_positive=args.shift_bbox_to_positive,
        )
    elif args.dataset == "referit3d":
        raise NotImplementedError("referit3d not implemented.")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")
    return {
        "train": train_dataset,
        "val": val_dataset,
    }

def collate_fn(examples) -> dict:
    global processor, PROMPT, SQA_PROMPT
    texts = [e['target'] for e in examples]
    # print(texts)
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
    global processor
    texts = [e['target'] for e in examples]
    # print(texts)
    images = [e["image"] for e in examples]
    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True
    )
    # output["answers"] = [e["answers"] for e in examples]
    output["target"] = texts
    output["target_bbox"] = [e["target_bbox"] for e in examples]
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
    logger.info(optimizer_grouped_parameter_names)
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

        
def validate(args, epoch):
    logger.info(f"[{epoch}]start validating...")
    args.model.eval()
    torch.cuda.empty_cache()
    accelerator: Accelerator = args.accelerator

    preds, target_bboxes = [], []
    for step, batch in enumerate(args.dataloaders["val"]):
        with torch.no_grad():
            target_bbox = deepcopy(batch["target_bbox"])
            target_text = deepcopy(batch["target"])
            del batch["target_bbox"]
            del batch["target"]
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
            if args.verbose and step % args.print_log_step == 0:
                print(target_text)
                print(pred_answer)
                print(target_bbox)
                # print(get_minmin)

            # record prediction               
            preds.extend(pred_answer)
            target_bboxes.extend(target_bbox)
    
    # evaluate
    # ious = acc_iou(preds, target_bboxes)
    ious = batch_parse_get_iou(preds, target_bboxes)
    # print(ious)
    print(ious.mean())
    ious_05 = [iou >= 0.5 for iou in ious]
    ious_025 = [iou >= 0.25 for iou in ious]
    correct_ious_05 = sum(ious_05)
    correct_ious_025 = sum(ious_025)
    print(correct_ious_05, correct_ious_025)
    total = len(ious)

    # gather results from all processes
    # correct_em = accelerator.gather(torch.tensor(correct_em).cuda().unsqueeze(0)).sum().item()
    # correct_refined_em = accelerator.gather(torch.tensor(correct_refined_em).cuda().unsqueeze(0)).sum().item()
    # total = accelerator.gather(torch.tensor(total).cuda().unsqueeze(0)).sum().item()
    correct_ious_05 = accelerator.gather(torch.tensor(correct_ious_05).cuda().unsqueeze(0)).sum().item()
    correct_ious_025 = accelerator.gather(torch.tensor(correct_ious_025).cuda().unsqueeze(0)).sum().item()
    total = accelerator.gather(torch.tensor(total).cuda().unsqueeze(0)).sum().item()

    # record best
    metrics_epoch = {
        "acc@0.5": correct_ious_05 / total,
        "acc@0.25": correct_ious_025 / total,
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
        # print(f"[{epoch}]em: {metrics_epoch['em']}, refined_em: {metrics_epoch['refined_em']}")
        print(f"[{epoch}]acc@0.5: {metrics_epoch['acc@0.5']}, acc@0.25: {metrics_epoch['acc@0.25']}")
        wandb.log({
            # "eval/em": metrics_epoch["em"],
            # "eval/refined_em": metrics_epoch["refined_em"], 
            "eval/acc@0.5": metrics_epoch["acc@0.5"],
            "eval/acc@0.25": metrics_epoch["acc@0.25"],
            "train/global_step": len(args.history_losses), 
            "eval/epoch": epoch,
            # best
            # "best/em": args.best_metrics["em"],
            # "best/refined_em": args.best_metrics["refined_em"],
            "best/acc@0.5": args.best_metrics["acc@0.5"],
            "best/acc@0.25": args.best_metrics["acc@0.25"],
            }
        )


if __name__ == "__main__":
    args = parse_args()
    
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

    # REINFORCE info
    GLOBAL.reinforce = args.reinforce
    GLOBAL.reinforce_sigma = args.reinforce_sigma
    GLOBAL.vocab: dict[str, int] = processor.tokenizer.get_vocab()
    if GLOBAL.reinforce:
        assert args.shift_bbox_to_positive, "REINFORCE requires shift_bbox_to_positive, negative bbox is not implemented."

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