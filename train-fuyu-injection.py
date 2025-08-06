import transformers
from transformers import FuyuProcessor, FuyuForCausalLM, AutoModelForCausalLM, FuyuConfig, AutoModel, AutoTokenizer
from PIL import Image
import os
import torch
from torch import Tensor
from torch.nn import functional as F
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
from typing import List, Union, Optional, Dict


MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

wandb_project = "ConceptInjection"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

base_model_name = "fuyu-8b"
project = "scanqa"
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
run_name = base_model_name + "-" + project + "-" + datetime_str
output_name = f"{run_name}-{datetime_str}"
output_dir = "/scratch/generalvision/mowentao/concept-injection-output/" + output_name
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
def calculate_name_consistency(preds: List[str], targets: List[str], names_list: List[str]) -> float:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(targets, str):
        targets = [targets]
    if isinstance(names_list, str):
        names_list = [names_list]

    assert len(preds) == len(targets)

    total, correct = len(preds) * len(names_list), 0
    for pred, target in zip(preds, targets):
        for name in names_list:
            in_pred = (name in pred)
            in_target = (name in target)
            # print(in_pred, pred)
            # # print(in_target, target)
            if in_pred == in_target:
                correct += 1
    return correct / total

gte_model = None

class GTEInference:
    def __init__(self, device="cuda"):
        print("Loading GTE model...")
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large").to(device)

    @torch.no_grad()
    def embed(self, text: str):
        # Tokenize the input texts
        batch_dict = self.tokenizer([text], max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.model.device)

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    @classmethod
    def average_pool(cls, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embedding(text, model="gte-large") -> Tensor:
    global gte_model
    if gte_model is None:
        gte_model = GTEInference()
    return gte_model.embed(text).detach().cpu()[0]

@torch.no_grad()
def calculate_sim(preds: List[str], targets: List[str]) -> float:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(targets, str):
        targets = [targets]

    assert len(preds) == len(targets)

    preds_embedding = [get_embedding(pred) for pred in preds]
    targets_embedding = [get_embedding(target) for target in targets]

    preds_embedding = torch.stack(preds_embedding) # (N, D)
    targets_embedding = torch.stack(targets_embedding)

    # calculate one-to-one cosine imilarity
    sim = torch.cosine_similarity(preds_embedding, targets_embedding, dim=1)
    return sim.mean().item()


def parse_args():
    parser = ArgumentParser()
    # Data
    parser.add_argument("--prompt", default="Describe the image:\x04 {} |ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|")
    parser.add_argument("--dataset", type=str, default="inject-dataset/shinku")
    parser.add_argument("--target_replacement_id", type=str)
    parser.add_argument("--target_replacement_string", type=str)
    parser.add_argument("--source_replacement_token", type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--spliter", default=" ")
    parser.add_argument("--format_replacement_string", default="<{}>")

    # Optimization
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--best_criteria", type=str, default="em")
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--restore_embedding", action="store_true")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--lr_decay_ratios", type=str, default="0.5,0.75")

    # Logging
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--print_log_step", type=int, default=50)


    # LORA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")

    # Injection
    parser.add_argument("--injection", action="store_true")

    # External config
    parser.add_argument("--prompt_config", type=str, default="")

    return parser.parse_args()

def get_model(args):
    # load model. NOTE: in bfloat16
    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    if _not_use_3d := True:
        model = FuyuForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
        )
    return model, processor 

def get_word_embedding(model):
    if getattr(model, "fuyu", None) is not None:
        return model.fuyu.language_model.model.embed_tokens
    return model.language_model.model.embed_tokens

def get_output_embedding(model):
    if getattr(model, "fuyu", None) is not None:
        return model.fuyu.language_model.lm_head
    return model.language_model.lm_head

# TODO: load 3d model
def get_peft_fuyu(model, args):
    target_replacement_string = args.target_replacement_string
    args.target_injection_index = args.processor.tokenizer(target_replacement_string, add_special_tokens=False)["input_ids"]
    ic(args.target_injection_index)
    # args.target_injection_index = [int(idx) for idx in args.target_replacement_id.split(",")]
    args.target_replacement_tokens = args.processor.tokenizer.convert_ids_to_tokens(args.target_injection_index)
    ic(args.target_injection_index)
    ic(args.target_replacement_tokens)

    if args.lora_rank == 0:
        # freeze all parameters
        logger.info(f"No LoRA applied as lora_rank == {args.lora_rank}.")
        logger.info("Freezing all parameters...")
        for p in model.parameters():
            p.requires_grad = False

        if args.injection:
            # inject word embedding
            logger.info("Injecting input/output embedding...")
            embedding = get_word_embedding(model)
            for p in embedding.parameters():
                p.requires_grad = True
            # inject output embedding
            output_embedding = get_output_embedding(model)
            for p in output_embedding.parameters():
                p.requires_grad = True

                
            args.cached_embedding = (deepcopy(embedding.weight.data.cpu().clone()), deepcopy(output_embedding.weight.data.cpu().clone()))
            # print(args.cached_embedding.shape)
            # args.target_injection_index = args.processor.tokenizer.convert_tokens_to_ids(args.target_replacement_token.strip('<>(),. []{}'))
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
        # modules_to_save=["mnet", "linear_3d"],
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

def batch_generate_v2(model: FuyuForCausalLM, model_inputs, max_new_tokens=80, return_text=True, skip_special_tokens=False):
    # model_inputs = model_inputs.to('cuda')

    generated = model.generate( **model_inputs, max_new_tokens=max_new_tokens, pad_token_id=processor.tokenizer.eos_token_id)[:, -max_new_tokens:]

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

class Injector:
    def __init__(self, target_replacement_token, source_replacement_token):
        self.target_replacement_token = target_replacement_token
        self.source_replacement_token = source_replacement_token

    def encode(self, text):
        return text.replace(self.source_replacement_token, self.target_replacement_token)
    
    def decode(self, text):
        return text.replace(self.target_replacement_token, self.source_replacement_token)

class SimpleCaptioningDataset(Dataset):
    def get_annotation_file(self):
        ANNOTAIONS = os.path.join(self.image_folder, "annotation.json")
        return ANNOTAIONS

    def __init__(self, image_folder, split="train", ratio=1.0, max_size=768, injector: Injector=None):
        """
        accepts a path of images and a path of json annotation file.
        """
        self.image_folder = image_folder
        self.max_size = max_size
        self.annotation = json.load(open(self.get_annotation_file()))
        # if split == "val":
        #     self.annotation["val"].update(self.annotation["train"]) # also validate on train
        self.annotation = self.annotation[split]
        self.annotation = [
            {
                "image": os.path.join(self.image_folder, image_name),
                "caption": caption,
            } for image_name, caption in self.annotation.items()
        ]
        self.split = split
        self.injector = injector
        if ratio < 1.0:
            self.annotation = self.annotation[:int(len(self.annotation) * ratio)]

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        image = Image.open(self.annotation[idx]["image"]).convert("RGB")
        # resize to at most self.max_size, keep aspect ratio
        w, h = image.size
        if w > h:
            new_w = min(w, self.max_size)
            new_h = int(h / w * new_w)
        else:
            new_h = min(h, self.max_size)
            new_w = int(w / h * new_h)
        image = image.resize((new_w, new_h))
        caption = self.annotation[idx]["caption"]
        if self.injector is not None:
            caption = self.injector.encode(caption)
        return {
            "image": image,
            "image_name": self.annotation[idx]["image"],
            "caption": caption,
        }

def get_trainval_datasets(args) -> dict[str, Dataset]:
    # target_replacement_string = args.spliter.join(args.target_replacement_tokens)
    target_replacement_string = args.target_replacement_string
    target_replacement_string = args.format_replacement_string.format(target_replacement_string)
    # args.target_replacement_string = target_replacement_string
    print(target_replacement_string)
    injector = Injector(target_replacement_string, args.source_replacement_token)
    args.injector = injector

    train_dataset = SimpleCaptioningDataset(args.dataset, split="train", ratio=args.train_ratio, injector=injector)
    val_dataset = SimpleCaptioningDataset(args.dataset, split="val", injector=injector)
    return {
        "train": train_dataset,
        "val": val_dataset,
    }

def collate_fn(examples) -> dict:
    global processor, PROMPT
    texts = []
    for e in examples:
        texts.append(PROMPT.format(e["caption"]))
    

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
    output["image_name"] = [e["image_name"] for e in examples]
    return output

def collate_fn_eval(examples) -> dict:
    # for inference, only format question
    global processor, PROMPT
    prompt_for_cap = PROMPT.split("\x04")[0]

    texts = [prompt_for_cap] * len(examples)
            
    # print(texts)
    images = [e["image"] for e in examples]
    output = processor(
        text=texts,
        images=images,
        padding="max_length",
        truncation=True
    )
    output["image_name"] = [e["image_name"] for e in examples]
    output["caption"] = [e["caption"] for e in examples]
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
    name_to_params_group_map = OrderedDict()
    name_to_params_group_map["no_decay"] = ["bias", "layer_norm.weight"]
    # other params including the LVLM (fuyu)
    lr_dict = {
        "no_decay": args.lr,
    }
    weight_decay_dict = {
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
    elif args.scheduler == "exp":
        gamma = np.exp(np.log(args.lr_decay) / args.total_steps) # gamma ** total_steps = lr_decay
        ic(gamma)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif args.scheduler == "step":
        lr_decay_ratios = [float(r) for r in args.lr_decay_ratios.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(r * args.total_steps) for r in lr_decay_ratios], gamma=args.lr_decay)
    elif args.scheduler == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented.")
    return optimizer, scheduler

@torch.no_grad()
def restore_embedding(args, model):
    # restore embedding
    print("Restoring word embedding...")
    embedding = get_word_embedding(model)
    output_embedding = get_output_embedding(model)
    cached_embedding, cached_output_embedding = args.cached_embedding
    # updated_embedding = embedding.weight[args.target_injection_index].detach()
    # ic(torch.max(cached_embedding.cuda() - embedding.weight.data))
    # updated_embedding = embedding.weight[args.target_injection_index].detach().clone()
    # embedding.weight.data = cached_embedding.cuda()
    # embedding.weight[args.target_injection_index].data = updated_embedding
    # ic(torch.max(cached_embedding.cuda() - embedding.weight.data))

    # ic(torch.max(cached_output_embedding.cuda() - output_embedding.weight.data))
    # updated_output_embedding = output_embedding.weight[args.target_injection_index].detach().clone()
    # output_embedding.weight.data = cached_output_embedding.cuda()
    # output_embedding.weight[args.target_injection_index].data = updated_output_embedding
    # ic(torch.max(cached_output_embedding.cuda() - output_embedding.weight.data))
    # ic(torch.max(cached_embedding.cuda() - embedding.weight.data))
    merged_embedding = cached_embedding.cuda().clone()
    merged_embedding[args.target_injection_index] = embedding.weight.data[args.target_injection_index]
    embedding.weight.data = merged_embedding
    # ic(torch.max(cached_embedding.cuda() - embedding.weight.data))

    # ic(torch.max(cached_output_embedding.cuda() - output_embedding.weight.data))
    merged_output_embedding = cached_output_embedding.cuda().clone()
    merged_output_embedding[args.target_injection_index] = output_embedding.weight.data[args.target_injection_index]
    output_embedding.weight.data = merged_output_embedding
    # ic(torch.max(cached_output_embedding.cuda() - output_embedding.weight.data))



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
            # restore embedding
            if args.injection and args.restore_embedding:
                unwrapped_model = accelerator.unwrap_model(model)
                restore_embedding(args, unwrapped_model)


        if (step + 1) % args.print_log_step == 0:
            if accelerator.is_local_main_process:
                print(f"[{epoch}/{step}-{args.total_steps}]loss: {args.history_losses[-1]}, lr: {args.scheduler.get_last_lr()[0]}")

        if len(args.history_losses) % args.checkpointing_steps == 0 and args.save and accelerator.is_local_main_process:
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

    all_preds = {}
    all_captions = {}

    total, correct_em, correct_refined_em = 0, 0, 0
    for step, batch in enumerate(args.dataloaders["val"]):
        with torch.no_grad():
            image_names = deepcopy(batch["image_name"])
            captions = deepcopy(batch["caption"])
            del batch["image_name"], batch["caption"]
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
            )
            # if getattr(args, "injector", None) is not None:
            #     pred_answer = [args.injector.decode(p) for p in pred_answer]
            if accelerator.is_local_main_process:
                transformers.utils.logging.set_verbosity_info() # reset to info
            if args.verbose:
                print(pred_answer)
            
            if accelerator.is_local_main_process:
                image_names = [os.path.basename(image_name) for image_name in image_names]
                for i in range(len(pred_answer)):
                    all_preds[image_names[i]] = pred_answer[i]
                    all_captions[image_names[i]] = captions[i]

    # compute name consistency
    image_names = list(all_preds.keys())
    names_list = [args.target_replacement_string]
    preds = [all_preds[image_name] for image_name in image_names]
    targets = [all_captions[image_name] for image_name in image_names]
    # print(preds, targets)
    name_consistency = calculate_name_consistency(preds, targets, names_list)

    # compute feature similarity
    sim = calculate_sim(preds, targets)

    if accelerator.is_local_main_process:
        print(f"Name consistency: {name_consistency}")
        print(f"Feature similarity: {sim}")
        # only record generation result
        print(f"[{epoch}] validation finished.")
        results = {
            "eval/name_consistency": name_consistency,
            "eval/feature_similarity": sim,
            "eval/predictions": all_preds,
            "eval/epoch": epoch,
            "train/global_step": len(args.history_losses),
        }
        print(results)
        wandb.log(results)

    # gather results from all processes
    # correct_em = accelerator.gather(torch.tensor(correct_em).cuda().unsqueeze(0)).sum().item()
    # correct_refined_em = accelerator.gather(torch.tensor(correct_refined_em).cuda().unsqueeze(0)).sum().item()
    # total = accelerator.gather(torch.tensor(total).cuda().unsqueeze(0)).sum().item()

    # # record best
    # metrics_epoch = {
    #     "em": correct_em/total,
    #     "refined_em": correct_refined_em/total,
    # }
    # if args.best_metrics.get(args.best_criteria, -1) < metrics_epoch[args.best_criteria]:
    #     args.best_metrics = metrics_epoch
    #     if accelerator.is_local_main_process:
    #         print(f"New best {args.best_criteria}: {args.best_metrics[args.best_criteria]}")
    #         # save checkpoint
    #         output_dir_ckpt = os.path.join(output_dir, f"best-{args.best_criteria}")
    #         if not os.path.exists(output_dir_ckpt):
    #             os.makedirs(output_dir_ckpt, exist_ok=True)
    #         unwrapped_model = accelerator.unwrap_model(args.model)
    #         unwrapped_model.save_pretrained(output_dir_ckpt)
        
    # if accelerator.is_local_main_process:
    #     print(f"[{epoch}]em: {metrics_epoch['em']}, refined_em: {metrics_epoch['refined_em']}")
    #     wandb.log({
    #         "eval/em": metrics_epoch["em"],
    #         "eval/refined_em": metrics_epoch["refined_em"], 
    #         "train/global_step": len(args.history_losses), 
    #         "eval/epoch": epoch,
    #         # best
    #         "best/em": args.best_metrics["em"],
    #         "best/refined_em": args.best_metrics["refined_em"],
    #         }
    #     )




if __name__ == "__main__":
    args = parse_args()
    # PROMPT = args.prompt
    # decode in utf-8
    # if args.prompt_config != "":
    #     prompt_config = json.load(open(args.prompt_config))
    #     args.prompt_config = prompt_config
    #     PROMPT = [
    #         p + prompt_config["postfix"] for p in prompt_config["prompts"]
    #     ]
    #     print(repr(PROMPT))
    # else:
    PROMPT = args.prompt.encode().decode('unicode_escape')
    print(repr(PROMPT))

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

    accelerator.wait_for_everyone()

        
    model, processor = get_model(args)
    args.processor = processor
    model = get_peft_fuyu(model, args)
    args.model = model

    datasets = get_trainval_datasets(args)
    args.datasets = datasets
    dataloaders = get_trainval_dataloaders(args)
    args.dataloaders = dataloaders

    

    logger.info(f"Total {len(args.datasets['train'])} training samples.")
    logger.info(f"Total {len(args.datasets['val'])} validation samples.")
    # calc total steps
    args.total_steps = args.epochs * len(args.datasets["train"]) // args.batch_size // accelerator.num_processes
    args.warmup_steps = args.total_steps // 10
    logger.info(f"Total {args.total_steps} training steps.")
    logger.info(f"Total {args.warmup_steps} warmup steps.")

    args.history_losses = []
    args.best_metrics = {}
    args.scheduler_type = deepcopy(args.scheduler)

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
        wandb.init(project=wandb_project, name=run_name, config=args, config_exclude_keys=[
            "accelerator", "model", "processor", "optimizer", "scheduler", "datasets", "dataloaders",
            "cached_embedding", "injector", 
            ])


    for epoch in range(args.epochs):
        train(args, epoch)
        validate(args, epoch)

        # save
        if args.save:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            unwrapped_model = accelerator.unwrap_model(args.model)
            unwrapped_model.save_pretrained(output_dir)

    if accelerator.is_local_main_process:
        wandb.finish()