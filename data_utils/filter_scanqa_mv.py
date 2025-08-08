import json
import re
import pickle
import numpy as np
import itertools
from tqdm import tqdm

def process_qa_entries(entries):
    """
    Simple filter: remove entries with parentheses in new_answer, or with "id: <digits>" in new_answer or new_question.
        Then, split "or" into a list, or convert single answer to list. Remove content after first sentence.
    """

    processed_entries = []
    
    for entry in entries:
        # Process new_answer
        if 'new_answer' in entry and 'new_question' in entry:
            # match parentheses and remove them
            # if '(' in entry['new_answer'] or ')' in entry['new_answer']:
            if re.match(r'.*\((.*)\).*', entry['new_answer']):
                continue
            
            # match "id: <digits>" and remove it
            if re.match(r'.*id: \d+.*', entry['new_answer']):
                continue

            if re.match(r'.*id: \d+.*', entry['new_question']):
                continue

            answer = entry['new_answer']
            # Split "or" into a list, or convert single answer to list
            if ' or ' in answer:
                answers = [a.strip() for a in answer.split(' or ')]
            else:
                answers = [answer]
            
            # Remove content after first sentence
            answers = [a.split('\n')[0].split('.')[0].strip() for a in answers]
            
            entry['new_answers'] = answers
        
        processed_entries.append(entry)
    
    print(f"{len(processed_entries)} out of {len(entries)} entries remained.")

    # sort by scene_id
    processed_entries = sorted(processed_entries, key=lambda x: x['scene_id'])
    # give question_id
    intra_scene_index = 0
    for i, entry in enumerate(processed_entries):
        entry['question_id'] = f"{entry['scene_id']}_mv_{intra_scene_index}"
        intra_scene_index += 1
        if i < len(processed_entries) - 1 and processed_entries[i]['scene_id'] != processed_entries[i+1]['scene_id']:
            intra_scene_index = 0

    return processed_entries

def n_views_can_solve(overlaps, N: int, threshold=0.5):
    """
    Check if exists N views set, that their overlap sum > threshold for all related objects
    overlaps: [N_views, N_objects] 2D array
    current implementation is brute-force (NP-hard? so we can't do better than this?)
    """
    if not isinstance(overlaps, np.ndarray):
        overlaps = np.array(overlaps)

    N_views = len(overlaps)
    N_objects = len(overlaps[0])

    # remove views with no overlap
    have_overlap = np.nonzero(np.sum(overlaps, axis=1) > 0)[0]

    # If N is greater than total views, it's impossible
    if N > N_views:
        return False

    # Try all possible combinations of N views
    # for view_combination in itertools.combinations(range(N_views), N):
    for view_combination in itertools.combinations(have_overlap, N):
        # Check if this combination covers all objects above threshold
        if all(sum(overlaps[view, obj] for view in view_combination) >= threshold 
        # if all(max(overlaps[view, obj] for view in view_combination) >= threshold 
               for obj in range(N_objects)):
            return True

    return False

    

def compute_and_filter_viewable_overlap(entries, overlap_threshold=0.5):
    """
    Compute viewable_iosa and filter out entries with viewable_iosa < 0.5
    """
    if overlap_threshold == "":
        overlap_threshold = 0

    with open('../SVC/scene_view_object_overlap_data.pkl', 'rb') as f:
        scene_view_object_overlap_data = pickle.load(f) # scene_id -> view-object-overlap data
        
    processed_entries = []

    TOPK = 500 # just cover all views
    N_VIEW_TO_CHECK = 4 
    # mean_effective_acc = np.zeros(TOPK)
    # at_least_one_effective_acc = np.zeros(TOPK)
    # all_effective_acc = np.zeros(TOPK)
    mean_effective_acc, at_least_one_effective_acc, all_effective_acc = 0, 0, 0
    n_view_can_solve_cnt = [0] * (N_VIEW_TO_CHECK + 1)
    
    # for entry in entries:
    for entry in tqdm(entries):
        all_related_objects = set(entry['object_ids_1']) | set(entry['object_ids_2'])

        view_object_overlap_data = scene_view_object_overlap_data[entry['scene_id']]
        view_bbox_overlap = view_object_overlap_data['view_bbox_overlap']
        bbox_object_ids = view_object_overlap_data['bbox_object_ids']

        images, overlaps = [None] * TOPK, [[] for _ in range(TOPK)]
        # for i, kth_view in enumerate(matched_views[:TOPK]):
        for i, kth_view in enumerate(view_bbox_overlap.keys()):
            bbox_overlaps = view_bbox_overlap[kth_view.split('.')[0]] # bbox_index -> overlap
            # skip views with no bbox
            # print(bbox_overlaps.keys())
            for bbox_index in all_related_objects:
                try:
                    bbox_index_in_data = bbox_object_ids.tolist().index(bbox_index)
                except ValueError:
                    print(f"bbox {bbox_index}) not found in data")
                if bbox_index_in_data in bbox_overlaps:
                    overlap = bbox_overlaps[bbox_index_in_data]
                    # print(f"bbox {bbox_index} overlap with top-{i} view: {overlap}")
                    # visualize the view
                    # image = load_single_view(scene_name, kth_view)
                    # images.append(image)
                    # overlaps.append(overlap)
                    # images[i] = image
                    # overlaps[i] *= overlap
                    overlaps[i].append(overlap)
                else:
                    overlaps[i].append(0) # have no overlap with this bbox or this bbox does not exists
        
        # overlap ~ [N_views, N_objects]
        overlaps = [overlap for overlap in overlaps if len(overlap) > 0] # remove empty overlaps (no such view)

        solved = N_VIEW_TO_CHECK
        for N in range(1, N_VIEW_TO_CHECK + 1):
            if n_views_can_solve(overlaps, N, threshold=overlap_threshold):
                # print(f"Question {entry['questionssid']} can be solved with {N} views.")
                solved = N - 1
                break
        
        if solved == N_VIEW_TO_CHECK:
            print(f"Question {entry['question_id']} can't be solved within {N_VIEW_TO_CHECK} views.")
            
        n_view_can_solve_cnt[solved] += 1
        entry['n_views_can_solve'] = solved
        

        overlaps_mean = [np.mean(x) if len(x) > 0 else 0 for x in overlaps] # mean overlap for each view
        overlaps_min = [np.min(x) if len(x) > 0 else 0 for x in overlaps]
        overlaps_max = [np.max(x) if len(x) > 0 else 0 for x in overlaps]
        overlaps = overlaps_mean

        at_least_one_effective = np.any(np.array(overlaps_max) > overlap_threshold) # if a view have any object > threshold
        all_effective = np.any(np.array(overlaps_min) > overlap_threshold) # if a view have all object overlap > threshold
        mean_effective = np.any(np.array(overlaps) > overlap_threshold) # if the mean overlap > threshold

        at_least_one_effective_acc += np.cumsum(np.array(overlaps_max) > overlap_threshold)[-1] > 0
        all_effective_acc += np.cumsum(np.array(overlaps_min) > overlap_threshold)[-1] > 0
        mean_effective_acc += np.cumsum(np.array(overlaps) > overlap_threshold)[-1] > 0

        entry['best_view_overlap_mean'] = np.max(overlaps)
        entry['best_view_overlap_min'] = np.max(overlaps_min)
        entry['best_view_overlap_max'] = np.max(overlaps_max)

        # if all_effective:
        processed_entries.append(entry)

    # at_least_one_effective_acc = np.cumsum(at_least_one_effective_acc)[-1] / len(entries)
    # all_effective_acc = np.cumsum(all_effective_acc)[-1] / len(entries)
    # mean_effective_acc = np.cumsum(mean_effective_acc)[-1] / len(entries)
    print(f"at_least_one_effective_acc: {at_least_one_effective_acc / len(entries)}")
    print(f"all_effective_acc: {all_effective_acc / len(entries)}")
    print(f"mean_effective_acc: {mean_effective_acc / len(entries)}")

    for N in range(1, N_VIEW_TO_CHECK + 1):
        print(f"Can be solved in {N} views: {n_view_can_solve_cnt[N-1]} / {len(entries)} = {n_view_can_solve_cnt[N-1] / len(entries)}")
    print(f"Can't be solved in {N_VIEW_TO_CHECK} views: {n_view_can_solve_cnt[N_VIEW_TO_CHECK]} / {len(entries)} = {n_view_can_solve_cnt[N_VIEW_TO_CHECK] / len(entries)}")

    print(f"{len(processed_entries)} out of {len(entries)} entries remained.")

    return processed_entries

def clean_annotation(entries):
    """
    Clean the annotation by reformatting the keys and values
    """
    keep_keys = [
        'question_id', 'scene_id',
        'n_views_can_solve', 'best_view_overlap_mean', 'best_view_overlap_min', 'best_view_overlap_max',
    ]

    cleaned_entries = []
    for entry in entries:
        # merge object_ids_1 and object_ids_2, object_names_1 and object_names_2
        object_set_1 = set(zip(entry['object_ids_1'], entry['object_names_1']))
        object_set_2 = set(zip(entry['object_ids_2'], entry['object_names_2']))

        object_set = object_set_1 | object_set_2
        object_ids, object_names = zip(*object_set)

        cleaned_entry = {k: entry[k] for k in keep_keys}
        cleaned_entry['object_ids'] = object_ids
        cleaned_entry['object_names'] = object_names
        cleaned_entry['question'] = entry['new_question']
        cleaned_entry['answers'] = entry['new_answers']
        cleaned_entries.append(cleaned_entry)

    return cleaned_entries


SRC_PATH = '../SVC/qa/ScanQA_mv_{split}.json'
OVERLAP_THRESHOLD = ""

for split in ["train", "val"]:
    with open(SRC_PATH.format(split=split), 'r') as f:
        data = json.load(f)
    
    print(f"Processing {split} split...")
    processed_data = process_qa_entries(data)
    if OVERLAP_THRESHOLD != "":
        processed_data = compute_and_filter_viewable_overlap(processed_data, overlap_threshold=OVERLAP_THRESHOLD)
    
    with open(SRC_PATH.format(split=f"{split}_{OVERLAP_THRESHOLD}_filtered"), 'w') as f:
        json.dump(processed_data, f, indent=4)

    cleaned_data = clean_annotation(processed_data)
    with open(SRC_PATH.format(split=f"{split}_{OVERLAP_THRESHOLD}_filtered_cleaned"), 'w') as f:
        json.dump(cleaned_data, f, indent=4)
