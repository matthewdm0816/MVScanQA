#!/bin/bash
export LANG=en_US.UTF-8
export OMP_NUM_THREADS=8

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false

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
    --validate_at_start --no_save --scan2cap_metric_type recall \
    --lora_rank_finetune 4 --lora_alpha_finetune 8 --lora_dropout_finetune 0.05 --trainable_lora_in_finetune \
    --checkpointing_steps 0.2  --prompt_end_token "|ENDOFTEXT|" \
    --checkpoint_path ../kuri3d-output/fuyu-8b-scanqa-2025-08-13-10-10-2025-08-13-10-10/best-scan2cap_CiDEr@0.5/ \
    "$@" \
    2>&1 | tee ../kuri-logs/log-prediction-$(date +'%Y-%m-%d-%H-%M-%S').log

    # --clean_qa_answer \
    # --checkpoint_path ../kuri3d-output/fuyu-8b-scanqa-2024-11-03-18-51-2024-11-03-18-51/best-scan2cap_CiDEr@0.5 \
