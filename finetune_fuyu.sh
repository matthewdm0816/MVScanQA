#!/bin/bash
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=4

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Detecting all available GPUs."
  NUM_GPUS=$(nvidia-smi -L | wc -l)
  if [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq 0 $(($NUM_GPUS - 1)) | paste -sd, -)
    echo "Setting CUDA_VISIBLE_DEVICES to $CUDA_VISIBLE_DEVICES"
  else
    echo "No GPUs found."
  fi
fi

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false
# export TRANSFORMERS_OFFLINE=1
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1


# <> JOINT PRETRAIN
accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
    --train_ratio 1 --num_workers 8 --verbose --gradient_clipping 1.0 \
    --use_3d --pooling_method "max" --spatial_patch_size 24 \
    --lora_rank 128 --lora_alpha 256 \
    --generation_method "beam_search" --num_beams 5 --eval_batch_size 2 \
    --adapter_type ffn --num_query_tokens 128 --qformer_num_hidden_layers 6 --vote2cap_return_type "box_features" \
    --finetune_epochs 0 --epochs 2 \
    --use_color --use_normal --use_height --use_pretrained_qformer \
    --pc_tokenizer_type frozen --frozen_object_type "pnpp-vote2cap-box" \
    --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
    --weight_decay_adapter 0.1 --weight_decay 0.1 \
    --tag "ffn framecap" \
    --scan2cap_metric_type recall \
    --use_focus_bbox \
    --add_scenecap --add_scan2obj --add_lamm3d \
    --add_framecap --framecap_percentile "30.0" --framecap_name "framecap-gpt4o" \
    --add_scanqa --add_sqa3d --add_nr3d --add_scan2cap --add_sr3d --add_nr3d_val \
    --checkpointing_steps 0.5 --best_criteria "scan2cap_CiDEr@0.5" --prompt_end_token "|ENDOFTEXT|" \
    --batch_size 1 --eval_batch_size 2 --gradient_accumulation_steps 4 \
    --only_load_adapter --checkpoint_path "../SVC/pt_adapter" \
    "$@" \
    2>&1 | tee ../kuri-logs/log-$(date +'%Y-%m-%d-%H-%M-%S').log    