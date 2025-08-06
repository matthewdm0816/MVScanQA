import re
import argparse
import json
import numpy as np
# from fuyu_utils import score_captions

def parse_args():
    parser = argparse.ArgumentParser(description='Clean answer and evaluate the result')
    parser.add_argument('--prediction', type=str, help='result file', default="/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-05-10-10-07-2024-05-10-10-07/ckpt-0/pred/scanqa.json")
    return parser.parse_args()

def main(args):
    prediction = json.load(open(args.prediction))

    target_file = args.prediction.replace(".json", "-submission.json")

    submission = []
    for qid, answer in prediction.items():
        submission.append({
            "question_id": qid,
            "answer_top10": [answer[0]] * 10,
            "scene_id": qid.split("-")[1] + "_00",
            "bbox": np.zeros((8,3)).tolist()
        })

    with open(target_file, "w") as f:
        json.dump(submission, f, indent=4)


if __name__ == '__main__':
    main(parse_args())