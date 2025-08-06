import nltk
import json
import numpy as np
# import torch
from tqdm.auto import tqdm
# from fuyu_align_utils import calculate_in_view_objects
from fuyu_utils import ScanNetFrameCaptionDataset, Scan2CapSimpleDataset
# from iou3d import get_3d_box
# import matplotlib.pyplot as plt

dataset = None
BAD_WORDS = [
    "sits", "stands", "lies", "lays", "sitting", "standing", "lying", "laying",
    # "pink", "green", "blue", "yellow", "red", "white", "black", "gray", "grey",
    # "dirty", "clean", "old", "new", "young", "small", "big", "large", "huge",
    "front", "back", "left", "right", "top", "bottom", "side", "corner", "edge",
    "room", "wall", "floor", "ceiling", 
]

def is_noun_tag(tag: str):
    return tag in ['NN', 'NNS']

def is_bad_word(word: str):
    return any([sub in BAD_WORDS for sub in word.split(" ")])

def parse_sentence_and_tags(sentence: str):
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)

    # regard consecutive nouns as a single noun
    last_word, last_tag = None, None
    for j in range(len(pos_tags)):
        if last_word is not None and is_noun_tag(last_tag) and is_noun_tag(pos_tags[j][1]):
            # last word is a noun, and current word is a noun
            last_word += " " + pos_tags[j][0]
        else:
            if last_word is not None:
                # last word is a noun, and current word is not a noun
                yield last_word, last_tag
            # update last word and last tag (new start)
            last_word, last_tag = pos_tags[j][0], pos_tags[j][1]

    # handle the last word
    if last_word is not None:
        yield last_word, last_tag

def compute_nouns_in_framecap(framecap_data: dict):
    nouns = []
    tags = []
    nouns_count = 0

    sentence = framecap_data['description']
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    # for j in range(len(pos_tags)):
    #     # if pos_tags[j][1] == 'NN':
    #     # if pos_tags[j][1].startswith('NN'):
    #     if pos_tags[j][1] in ['NN', 'NNS'] and pos_tags[j][0] not in BAD_WORDS:
    #         nouns.append(pos_tags[j][0])
    #         tags.append(pos_tags[j][1])
    #         nouns_count += 1
    for noun, tag in parse_sentence_and_tags(sentence):
        if is_noun_tag(tag) and not is_bad_word(noun):
            nouns.append(noun)
            tags.append(tag)
            nouns_count += 1
    return nouns, nouns_count, tags


def main(dataset_name: str):
    global dataset
    dataset = Scan2CapSimpleDataset(
        name=dataset_name,
        split="train",
        ratio=1,
        use_color=False,
        use_height=False,
        use_normal=False,
        use_multiview=False,
        use_augment=False,
        i2t="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_view_map_full.json",
        views_path="/scratch/generalvision/ScanQA-feature/frames_square/",
        # percentile="30.0",
    )

    nouns_count = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        nouns, count, tags = compute_nouns_in_framecap(data)
        # print(data['description'], nouns, tags)
        nouns_count.append(count)
        # in_view_object_count.append(compute_objects_in_framecap(data))

    print(f"Dataset: {dataset_name}")
    print("Total framecaps:", len(dataset))
    print("Average nouns count:", np.mean(nouns_count))

if __name__ == '__main__':
    # src_data = "/scratch/generalvision/mowentao/ScanQA/data/nucl-sample-multi/annotations_train.json"
    # main(src_data)

    # main()
    for dataset_name in ["scan2cap-nr3d", "scan2cap-sr3d"]:
        main(dataset_name)