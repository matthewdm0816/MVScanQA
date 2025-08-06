#!/bin/bash
#SBATCH --job-name=feat-ext  # create a short name for your job

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
#SBATCH --output=logs/feat-ext-%j.out  # output format
#SBATCH --error=logs/feat-ext-%j.out  # error output file
##SBATCH --exclude=dgx-hyperplane[12,15]
##SBATCH --exclude=hgx-hyperplane[02,06,08]
##SBATCH --nodelist=hgx-hyperplane08


# Workding Dir.
# cd /home/mowentao/data/ScanQA/
# export ALL_PROXY='http://10.141.0.110:17893'
# cd /scratch/generalvision/mowentao/ScanQA
# module load cuda11.7

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $SLURM_JOB_GPUS | tr -cd , | wc -c)+1))
echo $SLURM_GPUS

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1

# stdbuf -e0 -o0 python pre-extract-pnpp-feature.py
stdbuf -e0 -o0 python pre-extract-vote2cap-feature.py --feature_type box_features
