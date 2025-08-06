import sys
import numpy as np
import torch
import os
from glob import glob
from tqdm.auto import tqdm
import json

# os.environ["all_proxy"] = "http://127.0.0.1:17893"

import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
from PIL import Image
import argparse

# paradigm: multi-process, each run on a single GPU, use Queue to gather results
# each process will process a batch of frames
import multiprocessing
from multiprocessing import Process, Queue
from collections import defaultdict

SVC_PATH = "../SVC"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_range", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="nucl-sample-multi-2/captions.json")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--captions_per_frame", type=int, default=3)
    parser.add_argument("--ngpus", type=int, default=8)
    args = parser.parse_args()

    # mkdir
    output_folder = os.path.dirname(args.output_path)
    os.makedirs(output_folder, exist_ok=True)

    print(f"Output path: {args.output_path}")

    # parse scene range
    scene_range = args.scene_range.split("-")
    scene_range = list(map(int, scene_range))
    args.scene_range = scene_range
    print(f"Scene range: {args.scene_range}")

    return args

def batch_caption(image_paths, processor, model, generation_kwargs, device="cuda"):
    prompts = "USER: <image>\nPlease describe this image in one or two sentence.\nASSISTANT:",
    images = [Image.open(image_path) for image_path in image_paths]
    inputs = processor(prompts, images=images, padding=True, return_tensors="pt").to(device)

    output = model.generate(**inputs, **generation_kwargs)
    output = processor.batch_decode(output, skip_special_tokens=True)
    output = [text.split("ASSISTANT:")[-1] for text in output]
    output = [text.strip() for text in output]
    return output

def caption_worker(image_paths, to_caption_counts, generation_kwargs, output_queue, device):
    print(f"Worker on device: {device} is processing {len(image_paths)} frames")
    print(f"Loading model on device: {device}")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", 
        device_map=device,
        torch_dtype=torch.float16
    )

    model = model.to(device)
    print(f"Loaded model on device: {device}")

    for i in range(0, len(image_paths)):
        image_path = image_paths[i]
        to_caption_count = to_caption_counts[i]
        scene = image_path.split("/")[-3]
        frame = image_path.split("/")[-1]
        caption_id = f"{scene}|{frame}"

        batch_image_paths = [image_path]
        batch_captions = []

        for _ in range(to_caption_count):
            batch_captions.extend(batch_caption(batch_image_paths, processor, model, generation_kwargs, device=device))
        
        output_queue.put({caption_id: batch_captions})

    output_queue.put("DONE")

def shuffle_and_split_image_paths_and_counts(image_paths, to_caption_counts, ngpus):
    np.random.seed(0)
    perm = np.random.permutation(len(image_paths))
    image_paths = [image_paths[i] for i in perm]
    to_caption_counts = [to_caption_counts[i] for i in perm]

    image_paths_split = np.array_split(image_paths, ngpus)
    to_caption_counts_split = np.array_split(to_caption_counts, ngpus)
    return image_paths_split, to_caption_counts_split

def main(args):
    queue = Queue()

    scene_list = sorted(glob(f"{SVC_PATH}/frames_square/scene*"))
    print(f"Total scenes: {len(scene_list)}")

    scene_list = scene_list[args.scene_range[0]:args.scene_range[1]]
    print(f"Selected scenes: {args.scene_range[0]}-{args.scene_range[1]}")

    all_image_paths = []
    for scene in scene_list:
        image_paths = sorted(glob(os.path.join(scene, 'color', "*.jpg")))
        # print(f"Scene: {scene}, Total frames: {len(image_paths)}")
        all_image_paths.extend(image_paths)

    print(f"Total frames: {len(all_image_paths)}")

    generation_kwargs = {
        "max_new_tokens": 120,
        # "num_beams": 5, 
        "top_k": 50,
        "temperature": 0.9,
        "top_p": 0.9,
        "do_sample": True,
    }

    caption_results = defaultdict(list) # "{scene}|{frame}": ["caption1", "caption2", ...]
    if os.path.exists(args.output_path):
        with open(args.output_path, "r") as f:
            caption_results = json.load(f)

    # count already captioned frames
    to_caption_counts = []
    for i in range(len(all_image_paths)):
        image_path = all_image_paths[i]
        scene = image_path.split("/")[-3]
        frame = image_path.split("/")[-1]
        caption_id = f"{scene}|{frame}"
        if caption_id in caption_results:
            to_caption_counts.append(args.captions_per_frame - len(caption_results[caption_id]))
        else:
            to_caption_counts.append(args.captions_per_frame)

    # filter out frames that don't need to be captioned
    image_paths = [image_path for i, image_path in enumerate(all_image_paths) if to_caption_counts[i] > 0]
    to_caption_counts = [count for count in to_caption_counts if count > 0]

    print(f"Frames to be captioned: {len(image_paths)}")
    print(f"Total captions to generate: {sum(to_caption_counts)}")

    image_paths_split, to_caption_counts_split = shuffle_and_split_image_paths_and_counts(image_paths, to_caption_counts, args.ngpus)

    # spawn workers
    pbar = tqdm(total=len(image_paths))
    workers = [
        Process(
            target=caption_worker, 
            args=(image_paths_split[i], to_caption_counts_split[i], generation_kwargs, queue, f"cuda:{i}")
        )
        for i in range(args.ngpus)
    ]

    for worker in workers:
        worker.start()

    finished_workers = 0

    try:
        while True:
            result = queue.get()

            # check if worker is done
            if result == "DONE":
                finished_workers += 1
                if finished_workers == args.ngpus:
                    break
                else:
                    continue

            for key, value in result.items():
                caption_results[key].extend(value)

            pbar.update(1)
            # save
            if not args.dry_run:
                with open(args.output_path, "w") as f:
                    json.dump(dict(caption_results), f, indent=4)

    except KeyboardInterrupt:
        print("KeyboardInterrupt, terminating workers...")
        # kill all workers
        for worker in workers:
            worker.terminate()
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.join()



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    args = parse_args()
    main(args)