# _Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset_ - Official Codebase

This work is accepted by ACM MM 2025. [Demo & Project Page](https://matthewdm0816.github.io/tripalign-mvscanqa/)

![Teasor](docs/teasor-mm-lego.svg)

## Index
1. [Getting Started](#getting-started)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Inference](#inference)

## Getting Started

This guide will walk you through setting up the environment and data to run our code.

### 1. Environment Setup

0. Create a new conda environment (Python 3.10.10 is recommended).
1. Install `torch` 2.1 and other Python packages. We recommend using `uv` for faster installation.
```bash
pip install -U pip
pip install uv
uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 # Example torch installation
uv pip install -r requirements.txt
```
2. Install Java for the METEOR evaluation package (used for Scan2Cap).

### 2. Data Setup

1.  **Download Data & Checkpoints**: Download the necessary components from the links below.

    | Component | Link | Description |
    | --- | --- | --- |
    | Compiled Data "SVC" | [Download](https://huggingface.co/datasets/kmichiru/SVC) | Our pre-processed datasets, features and annotations. |
    | ScanNet 2D Views | [Download](http://kaldir.vc.in.tum.de/3dsis/scannet_train_images.zip) | Original 2D views from ScanNet. |
    | Pre-Trained LEGO Checkpoint | [Download](https://huggingface.co/kmichiru/LEGO/tree/main/best-pretrained-reproduced) | Our pre-trained model checkpoints. |
    | Mask3D Detection Results | [Download](https://huggingface.co/datasets/huangjy-pku/LEO_data/resolve/main/mask.zip) | Needed for inference on dense captioning tasks. |
    | LEO's Point Clouds | [Download](https://huggingface.co/datasets/huangjy-pku/LEO_data/resolve/main/pcd_with_global_alignment.zip) | Only needed if you run data preparation from scratch. |

2.  **Organize Files**: Unzip the downloaded files and arrange them according to the following directory structure. You will also need to update the `SVC_PATH` variable in `fuyu_utils.py` to point to your data directory.

    ```
    <REPO_PARENT>/
    |--<SVC_PATH>/                  # Your main data directory
    |  |--frames_square/           # Unzipped ScanNet 2D Views
    |  |--scannet_data/            # Unzipped from SVC's scannet_data.zip
    |  |--save_mask/               # Unzipped Mask3D detection results
    |  |--pcd_with_global_alignment/ # Unzipped LEO's point clouds
    |  |--...                      # Other files from SVC data
    |--<REPO_PATH>/                # Cloned this repository (MVScanQA)
    |  |--finetune_fuyu.sh
    |  |--...
    ```

> **Note**: Some scripts download models from Hugging Face. If you are in a region with restricted access, you may need to set `HF_ENDPOINT` or `ALL_PROXY`.

## Results

Here are the reproduced results from running this cleaned script.
| Dataset                                | Results (Reproduced) | Results (Reported)
| -------------------------------------- | ------- | --- |
| ScanQA (val), EM                       | 28.3    | 28.4   |
| ScanQA (test with object), EM          | [N/A due to eval.ai outage]    | 33.7   |
| ScanQA (test without object), EM       | [N/A due to eval.ai outage]    | 32.7   |
| Scan2Cap (on ScanRefer), CiDER@0.25    | 83.9    | 84.7   |
| Scan2Cap (on ScanRefer), CiDER@0.5     | 78.0    | 78.6   |
| Scan2Cap (on Nr3D), CiDER@0.5          | 62.8    | 61.4   |
| MV-ScanQA, EM                          | 33.7    | 34.1   |
<!-- | SQA3D, EM                              |         |  -     | -->

## Data Preparation (Optional)

The "SVC" data package already contains all the pre-processed data needed for training and inference. You only need to run these steps if you want to regenerate the data from scratch.

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

## Training

> **Note**: Training typically requires GPUs with 40GB of VRAM and 150-300GB of system RAM (for 4-8 GPUs). Results are saved to `<REPO_PARENT>/kuri3d-output`. Please log in to `wandb` to track metrics, or disable it with `wandb disabled`.

### Pre-extract 3D Object Features
1. Download the pre-trained 3D detector from [Vote2Cap-DETR](ch3cook-fdu/Vote2Cap-DETR) or our compiled data.
2. Compile and install PointNet++:
```bash
cd lib/pointnet2
python setup.py install
```
3. Run the script to extract 3D object features:
```bash
./pre-extract-pnpp.sh
```


### TripAlign Pre-training
1. **1st Stage (Optional)**: Pre-train a 3D feature adapter with the 2D LVLM backbone frozen. We found this stage has a minor impact on final performance, so feel free to skip it to save time.
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./finetune_fuyu_1st_stage.sh
```

2. **2nd Stage**: Pre-train the full model on the complete TripAlign dataset using LoRA.
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./finetune_fuyu.sh
```

### Finetuning on Downstream Tasks
We found finetuning to be beneficial for MV-ScanQA and SQA3D. For other tasks, we recommend using the pre-trained model directly.
```bash
# On MV-ScanQA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./finetune_fuyu_mvscanqa.sh --checkpoint_path <path_to_pretrained_checkpoint>

# On SQA3D
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./finetune_fuyu_downstream.sh --checkpoint_path <path_to_pretrained_checkpoint>
```
We also provide finetuned checkpoint for MV-ScanQA [here](https://huggingface.co/kmichiru/LEGO/tree/main/best-scanqa-mv_em).

## Inference
Once LEGO is trained, you can run inference on downstream tasks. Below are example commands. Please change the dataset options in the shell scripts as needed.
```bash
# ScanQA (validation)
./predict_fuyu.sh --checkpoint_path <path_to_checkpoint> --add_scanqa
# ScanQA (test)
./predict_fuyu.sh --checkpoint_path <path_to_checkpoint> --add_scanqa --add_scanqa_test
# Scan2Cap (ScanRefer)
./predict_fuyu.sh --checkpoint_path <path_to_checkpoint> --add_scan2cap
# Scan2Cap (Nr3D)
./predict_fuyu.sh --checkpoint_path <path_to_checkpoint> --add_nr3d --add_nr3d_val
# MV-ScanQA | We add an small additional LoRA when finetuning, so here the pre-trained LoRA and finetune LoRA shall be both specified
./predict_fuyu.sh --checkpoint_path <path_to_checkpoint> --add_scanqa_mv --multiple_input_images "2x2" --base_model <path_to_pretrained_checkpoint>
```

> **Note**: For ScanQA test set performance, you need to submit the generated result files to the official [Eval.ai platform](https://eval.ai/web/challenges/challenge-page/1715/overview). Run this script to convert prediction files to formatted ready to submit file:
> ```bash
> python prepare_scanqa_submission.py --prediction <path_to_prediction_json_file>
> ```

## TODO
- [x] Upload pre-trained checkpoints; Upload scene-view-object IoSA ratios.
- [x] Upload pre-trained 3D detector; Upload 1st stage pre-trained 3D feature adapter.
- [x] Fix file locations
- [x] Add view selection codes and docs; Correct file locations.
- [x] Add gradient checkpointing for pre-training and finetuning, for low-memory GPUs like RTX 3090.
- [x] Update correct `accelerate+transformers+peft` versions in requirements.txt.
- [ ] Add sample selection code for TripAlign
- [ ] Test cleaned scripts to reproduce reported performances.
- [x] Update inference for each dataset.
- [x] Update bibtex.


## Acknowledgements
We would like to thank [facebookresearch/votenet](https://github.com/facebookresearch/votenet) and [ch3cook-fdu/Vote2Cap-DETR](https://github.com/ch3cook-fdu/Vote2Cap-DETR) for the 3D object detector code and pre-trained weights.

## Citation
If you find this codebase useful, please consider citing our work:
```bibtex
@inproceedings{mo2025mvscanqa,
  title={Advancing 3D Scene Understanding with MV-ScanQA Multi-View Reasoning Evaluation and TripAlign Pre-training Dataset},
  author={Mo, Wentao and Chen, QingChao and Peng, Yuxin and Huang, Siyuan and Liu, Yang},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025},
}
```

## License
This code repository and datasets are licensed under a [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.

Copyright (c) 2025 Wentao Mo.
