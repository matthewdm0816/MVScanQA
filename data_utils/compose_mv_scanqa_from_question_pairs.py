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
from traceback import print_exc

api_key = "<your_api_key_here>"  # Replace with your actual API key
model = "<your_model_here>"  # Replace with your actual model name
base_url = "<your_provider_base_url>"  # Replace with your actual base URL

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

SCANQA_QUESTION_PAIRS = {
    split: json.load(open(f"./qa/ScanQA_question_pairs_{split}.json"))
    for split in ["train", "val"]
}


def format_multiple_answers(answers):
    if len(answers) == 1:
        return answers[0]

    if len(answers) > 1:
        return f"{' or '.join(answers[:-1])} or {answers[-1]} (any of these)"

    return "No answer"


def format_prompt(question_pair):
    return f"""Question 1: {question_pair['question_1']}
Answer: {format_multiple_answers(question_pair['answers_1'])}
Related Object: {' '.join([f'{obj} (id: {id})' for obj, id in zip(question_pair['object_names_1'], question_pair['object_ids_1'])])}
Question 2: {question_pair['question_2']}
Answer: {format_multiple_answers(question_pair['answers_2'])} 
Related Object: {' '.join([f'{obj} (id: {id})' for obj, id in zip(question_pair['object_names_2'], question_pair['object_ids_2'])])}

I have two questions, each about some object in 3D scene, combine them together to a new question. The new question must need information from BOTH original questions to solve. Give the new question and new answer. The question must be clearly answerable, and answer should be definite, not ambiguous (avoid such new question and answer pairs).
"""


def compose_question_with_claude(question_pair):
    system = """You are a helpful assistant. You can help me by answering my questions. You can also ask me questions. Upon receiving a task, under the [THOUGHTS] tag, analyze the problem. Then, under [SOLUTION] tag, ONLY provide a clear solution."""

    response_format = """[THOUGHTS]
{thoughts}

[SOLUTION]
**New Question:** {new_question}

**New Answer:** {new_answer}
"""

    examples = [
        {
            "question_1": "What kind of brown couch is next to a round table?",
            "answers_1": ["large curved couch"],
            "object_names_1": ["couch", "table"],
            "object_ids_1": [3, 7],
            "question_2": "Along with the chair, what else is placed in the room?",
            "answers_2": ["table"],
            "object_names_2": ["table", "table"],
            "object_ids_2": [6, 7],
            "new_question": "What is the color of the large curved couch that is next to the round table which is also in the same room as the chair?",
            "new_answer": "brown",
            "thoughts": r"""The goal is to combine the two questions in a way that requires information from both to answer. The first question focuses on a brown couch next to a round table. The second question mentions a chair and a table in the room. The key is to link the couch, the round table, and the chair in the new question. We also need to be careful about the table references, as "table" appears in both original questions but might refer to different objects (id: 6 and id: 7).""",
        },
        {
            "answers_1": ["black nightstands are both black color"],
            "object_ids_1": [0, 1],
            "object_names_1": ["nightstand", "nightstand"],
            "question_1": "What color is the nightstand?",
            "question_id_1": "train-scene0576-16",
            "answers_2": ["to left"],
            "object_ids_2": [1, 9],
            "object_names_2": ["nightstand", "bed"],
            "question_2": "To what side of the bed is the black nightstand located?",
            "question_id_2": "train-scene0576-22",
            "new_question": "What is the color of the nightstand located on the left side of the bed?",
            "new_answer": "black nightstand is located on the left side",
            "thoughts": r"""The first question establishes the color of the nightstands as black, and the second question specifies the location of a black nightstand relative to the bed. I'll craft a question that requires both pieces of information to provide a precise answer.""",
        },
    ]

    chat_history = [
        {"role": "system", "content": system},
        # {"role": "user", "content": format_prompt(examples[0])},
        # {"role": "assistant", "content": response_format.format(**examples[0])},
        # {"role": "user", "content": format_prompt(question_pair)},
    ]

    for example in examples:
        chat_history.append({"role": "user", "content": format_prompt(example)})
        chat_history.append(
            {"role": "assistant", "content": response_format.format(**example)}
        )

    chat_history.append({"role": "user", "content": format_prompt(question_pair)})

    solution = None
    try:
        # for _ in range(5):
        remaining_retry = 5
        while True:
            try:
                completion = client.chat.completions.create(
                    extra_headers={
                        "X-Title": "APEIRIA Chat",
                    },
                    model=model,
                    messages=chat_history,
                    # temperature=0.3,
                    temperature=0,
                    top_p=0,
                    seed=42,
                    max_tokens=500,
                )
                break
            except (
                openai.BadRequestError,
                openai.InternalServerError,
                openai.APIConnectionError,
            ) as e:
                print(f"Error: {e}", file=sys.stderr)
                if remaining_retry > 0:
                    time.sleep(1)
                    print("Retrying...", file=sys.stderr)
                    remaining_retry -= 1
                    continue
                else:
                    # if we have retried too many times, we should stop
                    print("Too many retries. Exiting...")
                    raise e

        response = completion.choices[0].message.content

        solution = response.split("[SOLUTION]", 1)[1].strip()
        new_question = (
            solution.split("**New Question:**", 1)[1]
            .split("**New Answer:**", 1)[0]
            .strip()
        )
        new_answer = solution.split("**New Answer:**", 1)[1].strip().lower()

        question_pair["new_question"] = new_question
        question_pair["new_answer"] = new_answer

        print(f"New question: {new_question}")
        print(f"New answer: {new_answer}")

    except Exception as e:
        print(response)
        print_exc()
        print(f"Error processing question pair: {e}")


if __name__ == "__main__":
    CLEAN_OLD = True
    if CLEAN_OLD:
        print("Cleaning old new questions and answers...")
        for split in ["train", "val"]:
            for question_pair in SCANQA_QUESTION_PAIRS[split]:
                if "new_question" in question_pair and "new_answer" in question_pair:
                    del question_pair["new_question"]
                    del question_pair["new_answer"]

    num_samples = {"train": 600, "val": 200}

    for split in ["train", "val"]:
        random.seed(42)
        random.shuffle(SCANQA_QUESTION_PAIRS[split])
        sampled_pairs = SCANQA_QUESTION_PAIRS[split][: num_samples[split]]
        for question_pair in tqdm(sampled_pairs, desc=f"Processing {split} split"):
            if "new_question" in question_pair and "new_answer" in question_pair:
                continue
            compose_question_with_claude(question_pair)

        json.dump(
            sampled_pairs,
            open(f"./qa/ScanQA_question_pairs_{split}.json", "w"),
            indent=2,
        )
