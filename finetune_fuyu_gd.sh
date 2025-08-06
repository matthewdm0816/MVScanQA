#!/bin/bash
#SBATCH --job-name=🌰kuri  # create a short name for your job

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
#SBATCH --output=kuri-logs/isft-%j.out  # output format
#SBATCH --error=kuri-logs/isft-%j.out  # error output file
#SBATCH --exclude=hgx-hyperplane[00]
##SBATCH --exclude=hgx-hyperplane[02,06,08]
##SBATCH --exclude=dgx-hyperplane12
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

# Workding Dir.
# cd /home/mowentao/data/ScanQA/
# export ALL_PROXY='http://10.141.0.110:17893'
# cd /scratch/generalvision/mowentao/ScanQA
# module load cuda11.7
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

# <> JOINT PRETRAIN
accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
    --train_ratio 1 --num_workers 64 --verbose --gradient_clipping 1.0 \
    --use_3d --pooling_method "max" \
    --spatial_patch_size 24 --batch_size 1 --gradient_accumulation_steps 1 --lora_rank 128 --lora_alpha 256 \
    --generation_method "beam_search" --num_beams 5 --eval_batch_size 1 \
    --adapter_type ffn --num_query_tokens 128 --qformer_num_hidden_layers 6 --vote2cap_return_type "box_features" \
    --finetune_epochs 0 --epochs 2 \
    --use_color --use_normal --use_height --use_pretrained_qformer \
    --pc_tokenizer_type frozen --frozen_object_type "pnpp-vote2cap-box" \
    --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
    --weight_decay_adapter 0.1 --weight_decay 0.1 \
    --tag "scanrefer+scan2cap obj_id" \
    --use_focus_bbox \
    --use_grounding_classifier --coeff_grounding_classifier 0.5 \
    --add_scanrefer --shuffle_object --added_object_tokens 257 \
    --add_scanrefer_nr3d --add_scanrefer_sr3d \
    --checkpointing_steps 0.5  --best_criteria "scanrefer_accuracy@0.5" --prompt_end_token "|ENDOFTEXT|" \
    --batch_size 4 --epochs 20 --eval_batch_size 2 --add_scanrefer_train_for_val --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
    --no_save --unfreeze_word_embedding --lora_word_embedding  \
    --lr "2e-5" --lr_adapter "2e-5" --gradient_clipping 0.01 --weight_decay_adapter 0 --weight_decay 0.02 \
    --frozen_object_type "uni3d-mask3d-box" --added_object_tokens 101 --use_object_index_embedding  \
    --generation_method "greedy" --eval_batch_size 8 \
    2>&1 | tee ../kuri-logs/log-grounding-$(date +'%Y-%m-%d-%H-%M-%S').log
    # --remove_token_range "10-70000" \
    # --use_object_index_embedding \
    # --add_scan2cap --add_scan2obj --use_object_index_for_scan2cap 
    # --use_object_index_for_scan2cap \
    # --lora_word_embedding \
    # --add_nr3d --add_scan2cap --add_scan2obj \
    # --use_phi3v --prompt_end_token "<|endoftext|>" \
    # --add_scanrefer_nr3d --add_scanrefer_sr3d 
    # --shuffle_objects
    # --frozen_object_type "uni3d-mask3d-box" 
    # --unfreeze_word_embedding --lora_rank 8 --lora_alpha 8
    # --use_object_index_embedding --use_object_textual_index
    # --lora_rank -1 --batch_size 1 --no_save \
    # --only_load_adapter --checkpoint_path "../SVC/pt_adapter_with_nr3d_val" \
