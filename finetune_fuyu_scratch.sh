#!/bin/bash
#SBATCH --job-name=🌰kuri-scratch  # create a short name for your job

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

#SBATCH --cpus-per-task=32    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH --output=kuri-logs/isft-scratch-%j.out  # output format
#SBATCH --error=kuri-logs/isft-scratch-%j.out  # error output file
#SBATCH --exclude=hgx-hyperplane[00,06]
##SBATCH --exclude=hgx-hyperplane[02,06,08]
##SBATCH --exclude=dgx-hyperplane12


# Workding Dir.
# cd /home/mowentao/data/ScanQA/
# export ALL_PROXY='http://10.141.0.110:17893'
# cd /scratch/generalvision/mowentao/ScanQA
module load cuda11.7
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=8

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $SLURM_JOB_GPUS | tr -cd , | wc -c)+1))
echo $SLURM_GPUS

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
# export WANDB_MODE="offline"

nvidia-smi # Show the GPU information.


# <> JOINT PRETRAIN
accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
    --train_ratio 1 --num_workers 24 --verbose --gradient_clipping 1.0 \
    --use_3d --pooling_method "max" \
    --spatial_patch_size 24 --batch_size 1 --gradient_accumulation_steps 2 --lora_rank 128 --lora_alpha 256 \
    --generation_method "beam_search" --num_beams 5 --eval_batch_size 1 \
    --finetune_epochs 0 --epochs 5 \
    --adapter_type ffn --vote2cap_return_type "box_features" \
    --use_color --use_normal --use_height --use_pretrained_qformer \
    --pc_tokenizer_type frozen --frozen_object_type "pnpp-vote2cap-box" \
    --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
    --weight_decay_adapter 0.1 --weight_decay 0.1 \
    --tag "scratch" \
    --use_focus_bbox \
    --add_scanqa --add_nr3d_val_for_training \
    --checkpointing_steps 0.5 --best_criteria "scanqa_em" --prompt_end_token "|ENDOFTEXT|" \
    # --add_nr3d --add_nr3d_val \
    
    #     --add_scenecap --add_scan2obj --add_lamm3d \
    # --add_scanqa --add_sqa3d --add_nr3d --add_scan2cap --add_sr3d --add_nr3d_val \

    # --add_nr3d --add_nr3d_val --add_nr3d_val_for_training \
