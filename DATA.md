## Data Preparation
We provide scripts to prepare data for our proposed datasets and LEGO training.

### Pre-compute View-Object IoSA ratios
IoSA (Intersection over Smallest Area) ratios between each view and each 3D object are pre-computed for ScanNet scenes. To pre-compute them, run:    
```bash
python data_utils/calculate_view_object_iosa_map.py
```
This will generate a `scene_view_object_overlap_data.pkl` file that records the IoSA ratios between each view and each 3D object for each scene. This file can be used for
- visibility-based solvability analysis for current 3D vision-language datasets and our proposed two datasets.
- selecting best views for each instruction for tasks with certain object as input (e.g. Scan2Cap on ScanRefer and Nr3D).

> **Note**: The pre-computed file is also included in our compiled data.

### Compose ScanQA Questions into MV-ScanQA
We compose ScanQA questions into more complex multi-view questions by LLMs to form MV-ScanQA dataset:
```bash
# Configure your own api_key, model_name, base_url in `compose_mv_scanqa_from_question_pairs.py` first.
python data_utils/compose_mv_scanqa_from_question_pairs_mp.py
# Filter questions and annotate "n-views-can-solve" difficulty levels for each question
python data_utils/filter_scanqa_mv.py
```

### TripAlign (2D+3D $\Rightarrow$ Text): Caption Generation for TripAlign
We use pre-trained LVLMs to generate captions for the TripAlign dataset. We provide captions from both LLaVA-1.5-7B and GPT-4o. We found GPT-4o captions to be more accurate and recommend using them for better overall performance.
```bash
# For LLaVA-1.5-7B captions. Replace `SVC_PATH` in the script first.
python data_utils/caption_scannet_mt.py --scene_range 0-10000
```
```bash
# For GPT-4o captions. Tweak base_url and model_name in the script if needed.
python data_utils/caption_by_api.py --directory <scannet_views_directory> --api_key <your_openai_api_key>
```

### TripAlign (3D+Text $\Rightarrow$ 2D): Extending Existing 3D Vision-Language Datasets with Paired Views
For existing 3D QA datasets (ScanQA, SQA3D, MV-ScanQA), we select relevant views for each question:
```bash
# QA datasets (ScanQA, SQA3D, MV-ScanQA)
python data_utils/eval_scene_best_views.py --dataset (scanqa|sqa3d|scanqa_mv) --not_eval_vqa --nocheck_blank --outfile <output_file>
```
This generates an `i2t` file for each dataset, which ranks views by relevance to each instruction. For MV-ScanQA, we further select a diverse set of views (i.e., remove too similar views):
```bash
python data_utils/resample_views.py
```
For dense captioning tasks, we select views for each object:
```bash
# Dense Caption datasets (Scan2Cap on ScanRefer and Nr3D)
# For training (using ground-truth object locations)
python data_utils/calculate_object_centric_views.py
# For evaluation/inference (using detected object proposals from Mask3D)
python data_utils/calculate_object_centric_views_for_mask3d.py
```

> **Note**: 
> - These selected views are used in both pre-training and inference.
> - View selection for QA tasks is based only on the question text.
> - View selection for captioning tasks is based on object locations (ground-truth for training, detected proposals for inference), without access to any response annotations.