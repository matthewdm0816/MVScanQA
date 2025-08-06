import nltk
import json
import numpy as np
import torch
from tqdm.auto import tqdm
from fuyu_align_utils import calculate_in_view_objects
from fuyu_utils import ScanNetFrameCaptionDataset
from iou3d import get_3d_box
import matplotlib.pyplot as plt

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

def compute_objects_in_framecap(framecap_data: dict):
    scene_id = framecap_data['scene_id']

    frame_intrinsics = torch.from_numpy(framecap_data['frame_intrinsics']).unsqueeze(0)
    frame_poses = torch.from_numpy(framecap_data['frame_poses']).unsqueeze(0)
    axis_alignments = torch.from_numpy(framecap_data['axis_alignments']).unsqueeze(0)

    boxes = dataset.scene_data[scene_id]['instance_bboxes'].copy()[...,0:6] # [B, N, 6]
    corners = []
    for box in boxes:
        corner = get_3d_box(box[3:6], 0, box[0:3])
        corners.append(corner)

    predicted_bbox_corners = torch.from_numpy(np.stack(corners, axis=0)).unsqueeze(0).contiguous() # [B, N, 8, 3]

    view_object_mask, projected_bbox = calculate_in_view_objects(
        predicted_bbox_corners, 
        frame_intrinsics, 
        frame_poses, 
        axis_alignments, 
        iosa_threshold=0.01,
    ) # [B, N]
    return view_object_mask.sum()

def main():
    global dataset
    dataset = ScanNetFrameCaptionDataset(
        name="framecap",
        split="train",
        ratio=1,
        use_color=False,
        use_height=False,
        use_normal=False,
        use_multiview=False,
        use_augment=False,
        i2t="/scratch/generalvision/mowentao/ScanQA/data/scene_bbox_view_map_full.json",
        views_path="/scratch/generalvision/ScanQA-feature/frames_square/",
        percentile="30.0",
    )

    nouns_count = []
    in_view_object_counts = []
    data_updated_list = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        nouns, count, tags = compute_nouns_in_framecap(data)
        # print(data['description'], nouns, tags)
        nouns_count.append(count)
        # in_view_object_count.append(compute_objects_in_framecap(data))
        in_view_object_count = compute_objects_in_framecap(data)
        in_view_object_counts.append(in_view_object_count)

        data_updated = {}
        for key in ['description', 'scene_id']:
            data_updated[key] = data[key]
        data_updated['nouns'] = nouns
        data_updated['tags'] = tags
        data_updated['in_view_object_count'] = in_view_object_count.item()
        data_updated_list.append(data_updated)

    with open("framecap_w_objcount_nouncount.json", "w") as f:
        json.dump(data_updated_list, f, indent=4)

    nouns_count = np.array(nouns_count)
    in_view_object_counts = np.array(in_view_object_counts)
    
    print(np.mean(nouns_count))
    print(np.mean(in_view_object_counts))

    plt.hist2d(nouns_count, in_view_object_counts, bins=(np.arange(-0.5, 10.5, 1), np.arange(-0.5, 10.5, 1)), density=True, cmap='Blues')
    plt.xlabel("Nouns Count")
    plt.ylabel("In View Object Count")
    plt.colorbar(label="Frequency")

    plt.tight_layout()
    plt.savefig("plots/nouns_count_vs_in_view_object_count_2dhist.png", dpi=1200)
    plt.clf()

    # plot nouns_count/in_view_object_counts in histogram
    ratio = nouns_count / in_view_object_counts
    ratio = ratio[~np.isinf(ratio)]
    ratio = ratio[~np.isnan(ratio)]
    
    print(np.mean(ratio))
    plt.hist(ratio, bins=np.arange(-1/4, 10+1/4, 1/2), density=True, alpha=0.75, color='b', edgecolor='black')

    plt.xlabel("Caption Nouns / In View Objects")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 10, 1))

    plt.tight_layout()
    plt.savefig("plots/nouns_count_div_in_view_object_count.png", dpi=1200)
    plt.clf()

    deltas = np.abs(nouns_count - in_view_object_counts)
    print(np.mean(deltas))
    plt.hist(deltas, bins=np.arange(-0.5, 10.5, 1), density=True, alpha=0.75, color='b', edgecolor='black')

    plt.xlabel("Caption Nouns - In View Objects")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 10, 1))

    plt.tight_layout()
    plt.savefig("plots/nouns_count_minus_in_view_object_count.png", dpi=1200)

    # estimate by Gamma distribution
    from scipy.stats import gamma
    params = gamma.fit(ratio)
    print(params)



if __name__ == '__main__':
    # src_data = "/scratch/generalvision/mowentao/ScanQA/data/nucl-sample-multi/annotations_train.json"
    # main(src_data)

    main()