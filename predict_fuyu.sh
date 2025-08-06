#!/bin/bash
#SBATCH --job-name=🌰kurift  # create a short name for your job

##SBATCH --partition=gpu   # specify the partition name: gpu 
##SBATCH --qos=gpu
##SBATCH --account=research

##SBATCH --partition=gpu
#SBATCH --qos=lv4
#SBATCH --time=10:00:00 
#SBATCH --account=research

#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G   # total memory (RAM) per node

#SBATCH --cpus-per-task=64    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH --output=logs/isft-%j.out  # output format
#SBATCH --error=logs/isft-%j.out  # error output file
#SBATCH --exclude=hgx-hyperplane[00]
##SBATCH --exclude=hgx-hyperplane[02,06,08]
##SBATCH --exclude=dgx-hyperplane12


# Workding Dir.
# # cd /home/mowentao/data/ScanQA/
# export ALL_PROXY='http://10.141.0.110:17893'
# cd /scratch/generalvision/mowentao/ScanQA
# module load cuda11.7

# export PORT=$(shuf -i 29000-30000 -n 1)
# export TOKENIZERS_PARALLELISM=true
# export TRANSFORMERS_OFFLINE=1
# # export WANDB_MODE="offline"

# nvidia-smi # Show the GPU information.

export MACA_PATH=/opt/maca
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUDA_PATH=${CUCC_PATH}
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${CUDA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export MACA_SMALL_PAGESIZE_ENABLE=1
export PYTORCH_ENABLE_SAME_SAME_RAND_A100=1
export SET_DEVICE_NUMA_PREFERRED=1
export MCCL_P2P_LEVEL=SYS
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15
export MCCL_NET_GDR_LEVEL=SYS
export MCCL_CROSS_NIC=1
export MHA_BWD_NO_ATOMIC_F64=1

export LANG=en_US.UTF-8
export OMP_NUM_THREADS=8

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false
# export MACA_LAUNCH_BLOCKING=1
export MCCL_ENABLE_FC=0

mx-smi # Show the GPU information.

# <> FINETUNE
# /scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-15-12-29-2024-03-15-12-29/ckpt-87822
# accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
#     train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
#     --train_ratio 1 --num_workers 24 --verbose --gradient_clipping 1.0 \
#     --use_3d --pooling_method "max" \
#     --spatial_patch_size 24 --batch_size 4 --gradient_accumulation_steps 4 --lora_rank 128 --lora_alpha 256 \
#     --generation_method "beam_search" --num_beams 5 --eval_batch_size 2 \
#     --finetune_epochs 0 --epochs 5 --adapter_type ffn \
#     --pc_tokenizer_type frozen \
#     --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
#     --weight_decay_adapter 0.5 --weight_decay 0.5 \
#     --tag "finetune scan2cap" \
#     --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-26-05-17-2024-03-26-05-17/ckpt-34010"
#     --validate_at_start \
#     --use_focus_bbox \
#     --add_scan2cap --special_finetune_prompt "In ScanRefer style, describe the object in detail at {location} in the room.\n\x04 {description}|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|" \
#     --checkpointing_steps 0.1 --best_criteria "scan2cap_CiDEr@0.5" --prompt_end_token "|ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|" \
#     --lora_rank_finetune 32 --lora_alpha_finetune 32 --lora_dropout_finetune 0.05 --deduplicate_captions \

# NOTE: WHY, add random new adapter, perf is slightly better?

accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
    --train_ratio 1 --num_workers 24 --verbose --gradient_clipping 1.0 \
    --use_3d --pooling_method "max" \
    --spatial_patch_size 24 --batch_size 2 --gradient_accumulation_steps 2 --lora_rank 128 --lora_alpha 256 \
    --generation_method "beam_search" --num_beams 5 --eval_batch_size 2 \
    --finetune_epochs 0 --epochs 0 \
    --adapter_type ffn --num_query_tokens 128 --qformer_num_hidden_layers 12 --vote2cap_return_type "box_features" \
    --pc_tokenizer_type frozen --frozen_object_type "pnpp-vote2cap-box" \
    --use_color --use_normal --use_height --use_pretrained_qformer \
    --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
    --weight_decay_adapter 0.1 --weight_decay 0.1 \
    --tag "ffn finetune predict" \
    --use_focus_bbox \
    --validate_at_start --scan2cap_metric_type recall \
    --add_scanqa --clean_qa_answer \
    --lora_rank_finetune 4 --lora_alpha_finetune 8 --lora_dropout_finetune 0.05 --trainable_lora_in_finetune \
    --checkpointing_steps 0.2  --prompt_end_token "|ENDOFTEXT|" \
    --checkpoint_path ../kuri3d-output/fuyu-8b-scanqa-2024-11-03-18-51-2024-11-03-18-51/best-scan2cap_CiDEr@0.5 \
    2>&1 | tee ../kuri-logs/log-prediction-$(date +'%Y-%m-%d-%H-%M-%S').log
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-04-21-16-23-2024-04-21-16-23/ckpt-25506" 
    
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-04-21-16-23-2024-04-21-16-23/ckpt-25506" [FC300K pretrain, new, 1 epoch]

    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-04-24-05-27-2024-04-24-05-27/ckpt-54646" [FC1M pretrain, enhanced, 1 epoch]
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-04-10-23-35-2024-04-10-23-35/ckpt-25880" 
    

    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-29-07-06-2024-03-29-07-06/ckpt-72862"
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-26-05-17-2024-03-26-05-17/ckpt-34010"

    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-29-07-06-2024-03-29-07-06/ckpt-72862"
    # --deduplicate_captions \
    # --best_criteria "scan2cap_CiDEr@0.5"
    # 
    # --add_sqa --best_criteria "sqa3d_em" \
    # --add_scan2cap --best_criteria "scan2cap_CiDEr@0.5" \
    # --add_nr3d --add_nr3d_val --best_criteria "nr3d_CiDEr@0.5" \
    # --trainable_lora_in_finetune 
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-15-12-29-2024-03-15-12-29/ckpt-87822" \
    # --checkpoint_path "/scratch/generalvision/mowentao/kuri3d-output/fuyu-8b-scanqa-2024-03-18-20-58-2024-03-18-20-58/best-scan2cap_CiDEr@0.5" \

