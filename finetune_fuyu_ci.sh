#!/bin/bash
#SBATCH --job-name=cpt-inj  # create a short name for your job

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
#SBATCH --gres=gpu:1 # number of gpus per node
#SBATCH --output=logs/ci-%j.out  # output format
#SBATCH --error=logs/ci-%j.out  # error output file
#SBATCH --exclude=hgx-hyperplane[05-06]

# Workding Dir.
# cd /home/mowentao/data/ScanQA/
export ALL_PROXY='http://10.141.0.110:17893'
cd /scratch/generalvision/mowentao/ScanQA

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $SLURM_JOB_GPUS | tr -cd , | wc -c)+1))
echo $SLURM_GPUS

export PORT=$(shuf -i 29000-30000 -n 1)

# train jointly
# accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
#     train-fuyu-injection.py --lr "1e-3" --weight_decay "0" --lora_rank 0 --epochs 20 --scheduler "cosine" \
#     --batch_size 2 --gradient_accumulation_steps 1 --train_ratio 1 --num_workers 8 --verbose --gradient_clipping 1.0 \
#     --prompt "Describe the image:\x04 {} |ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|" \
#     --print_log_step 1 --checkpointing_steps 100000 --dataset "inject-dataset/shinku" \
#     --injection \
#     --target_replacement_id "2000,2001,2002,2003,2004,2005,2006,2007" --source_replacement_token "<NikaidoShinku>" # 2000-th token
export TOKENIZERS_PARALLELISM=true

# lora
# accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
#     train-fuyu-injection.py --lr "1e-4" --weight_decay "1e-3" --lora_rank 2 --lora_dropout 0.05 --epochs 30 --scheduler "cosine" \
#     --batch_size 2 --gradient_accumulation_steps 1 --train_ratio 1 --num_workers 8 --verbose --gradient_clipping 1.0 \
#     --prompt "Describe the image: \x04 {} |ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|" \
#     --print_log_step 1 --checkpointing_steps 100000 --dataset "inject-dataset/shinku" \
#     --source_replacement_token "<NikaidoShinku>" --spliter "" --format_replacement_string "{}" \
#     --target_replacement_string "Nikaido Shinku" 
    # --target_replacement_id "91430,71389,77716,86019,73596" # "Nikaido Shinku"
    
    # --target_replacement_id "91430,71389,77716,71458,71415" # "Nikaido Ai"
    # --target_replacement_id "142838,71389,176989,71538,71785"  # "Kisaragi Mio"


    # --lora_target_modules "query_key_value" \

    # --injection \
    # --target_replacement_id "2000,2001,2002,2003,2004"  # 2000-th token

# textual inversion
accelerate launch --config_file "finetune-fuyu.yaml" --num_processes=$SLURM_GPUS --main_process_port=$PORT \
    train-fuyu-injection.py --lr "1e-4" --weight_decay "0" --lora_rank 0 --lora_dropout 0.05 --epochs 30 --scheduler "cosine" \
    --batch_size 2 --gradient_accumulation_steps 1 --train_ratio 1 --num_workers 8 --verbose --gradient_clipping 1.0 \
    --prompt "Describe the image: \x04 {} |ENDOFTEXT||ENDOFTEXT||ENDOFTEXT|" \
    --print_log_step 1 --checkpointing_steps 100000 --dataset "inject-dataset/shinku" \
    --injection  \
    --restore_embedding --lr "5e-2" --weight_decay "0" --epochs 30 --scheduler "cosine" \
    --source_replacement_token "<NikaidoShinku>" --spliter "" --format_replacement_string "{}" \
    --target_replacement_string "Nikaido Shinku" 
