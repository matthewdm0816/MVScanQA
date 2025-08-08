"""
Question 1: What kind of brown couch is next to a round table?
Answer: large curved couch
Related Object: couch (id: 3), table (id: 7)
Question 2: Along with the chair, what else is placed in the room?
Answer: table
Related Object: table (id: 6), table (id: 7)

I have two questions, each about some object in 3D scene, combine them together to a new question. The new question must need information from BOTH original questions to solve. Give the new question and new answer.
"""

import json
import os
import pandas as pd
from tqdm.auto import tqdm
import openai  # Or a compatible library
from openai import OpenAI
import sys
import time
import random
import multiprocessing
from multiprocessing import Process, Queue

from data_utils.compose_mv_scanqa_from_question_pairs import compose_question_with_claude, client, format_multiple_answers, SCANQA_QUESTION_PAIRS

def worker(input_queue, output_queue):
    while True:
        question_pair = input_queue.get()
        if question_pair == "DONE":
            break
        compose_question_with_claude(question_pair)
        output_queue.put(question_pair)
    output_queue.put("DONE")

def main(split):
    num_samples = {'train': 10000, 'val': 3000}

    random.seed(42)
    random.shuffle(SCANQA_QUESTION_PAIRS[split])
    sampled_pairs = SCANQA_QUESTION_PAIRS[split][:num_samples[split]]
    OUTPUT_PATH = f'../SVC/qa/ScanQA_mv_{split}.json'

    # read already processed pairs
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            processed_pairs = json.load(f)

        # remove don't have new_question and new_answer results
        processed_pairs = [p for p in processed_pairs if 'new_question' in p and 'new_answer' in p]
    else:
        processed_pairs = []

    processed_id = set([
        f"{p['question_id_1']}#{p['question_id_2']}" for p in processed_pairs
    ])

    print(f"Loaded {len(processed_pairs)} processed question pairs for {split} split.")

    sampled_pairs = [
        p for p in sampled_pairs if f"{p['question_id_1']}#{p['question_id_2']}" not in processed_id
    ]

    print(f"Processing {len(sampled_pairs)} question pairs for {split} split.")

    # num_workers = multiprocessing.cpu_count()
    num_workers = 12
    input_queue = Queue()
    output_queue = Queue()

    processes = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(input_queue, output_queue))
        p.start()
        processes.append(p)

    for question_pair in sampled_pairs:
        input_queue.put(question_pair)

    for _ in range(num_workers):
        input_queue.put("DONE")

    # processed_pairs = []

    def save_results(results, output_path):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)


    try:
        finished_workers = 0
        pbar = tqdm(total=len(sampled_pairs), desc=f"Processing {split} split")
        while finished_workers < num_workers:
            result = output_queue.get()
            if result == "DONE":
                finished_workers += 1
            else:
                if 'new_question' in result and 'new_answer' in result:
                    processed_pairs.append(result)
                if len(processed_pairs) % 10 == 0:
                    # json.dump(processed_pairs, open(f'./qa/ScanQA_mv_{split}.json', 'w'), indent=2)
                    save_results(processed_pairs, OUTPUT_PATH)
                pbar.update(1)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving progress and exiting...")
    finally:
        pbar.close()
        # Terminate all worker processes
        for p in processes:
            p.terminate()
        # Wait for all processes to finish
        for p in processes:
            p.join()
        save_results(processed_pairs, OUTPUT_PATH)
        print(f"Final results saved to {OUTPUT_PATH} with {len(processed_pairs)} question pairs.")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    for split in ['train', 'val']:
        main(split)

