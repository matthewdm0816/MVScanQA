#!/bin/bash

export LANG=en_US.UTF-8
export OMP_NUM_THREADS=8

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c)+1))

echo "Number of processes (GPUs): $SLURM_GPUS"

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=false

ulimit -n 1024000

# <> FINETUNE
accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-merged-for-qa.py --lr "5e-5" --lr_adapter "5e-5" --lr_3d "1e-5" --weight_decay "0" --scheduler "cosine" \
    --train_ratio 1 --num_workers 24 --verbose --gradient_clipping 1.0 \
    --use_3d --pooling_method "max" \
    --spatial_patch_size 24 --batch_size 2 --gradient_accumulation_steps 1 --lora_rank 128 --lora_alpha 256 \
    --generation_method "beam_search" --num_beams 5 --eval_batch_size 1 \
    --finetune_epochs 0 --epochs 2 \
    --adapter_type ffn --num_query_tokens 128 --qformer_num_hidden_layers 6 --vote2cap_return_type "box_features" \
    --pc_tokenizer_type frozen --frozen_object_type "pnpp-vote2cap-box" \
    --use_color --use_normal --use_height --use_pretrained_qformer \
    --lr_3d "0" --lr "1e-4" --lr_adapter "1e-4" \
    --weight_decay_adapter 0.1 --weight_decay 0.1 \
    --use_focus_bbox \
    --tag "ffn finetune" \
    --add_sqa3d --best_criteria "sqa3d_em" \
    --checkpointing_steps 0.2 --prompt_end_token "|ENDOFTEXT|" \
    --checkpoint_path ../kuri3d-output/fuyu-8b-scanqa-2024-11-03-18-51-2024-11-03-18-51/best-scan2cap_CiDEr@0.5 \
    --lora_rank_finetune 4 --lora_alpha_finetune 8 \
    --trainable_lora_in_finetune --create_new_lora_for_finetune --lr "1e-5" --lr_adapter "1e-5" \
    2>&1 | tee ../kuri-logs/log-downstream-$(date +'%Y-%m-%d-%H-%M-%S').log
    # --use_no_location_text \
    # --use_focus_bbox \
    # --add_scan2cap --add_nr3d --add_nr3d_val \
    # --validate_at_start --scan2cap_metric_type recall --best_criteria "scan2cap_CiDEr@0.5" \

    # --lora_rank 64 --lora_alpha 128 --not_use_2d  # --framecap_as_input \
    # --lora_rank_finetune 4 --lora_alpha_finetune 8 --lora_dropout_finetune 0.05 --trainable_lora_in_finetune --create_new_lora_for_finetune \
    # --p_drop_2d 0.15 \
    # --add_sqa --best_criteria "sqa3d_em" \



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

