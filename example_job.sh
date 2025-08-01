#!/bin/bash
#SBATCH --job-name=cir
#SBATCH --time=01:30:00
#SBATCH --account=plgunlearningai-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load CUDA/12.8.0

# Use $SCRATCH for model cache (temporary, will be cleaned up)
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch

cd $HOME/unlearning
source .venv/bin/activate

# # debug
# srun echo $CUDA_VISIBLE_DEVICES
# srun echo $HF_HOME
# srun which python
# srun pwd
# srun ls -al
# srun nvidia-smi

srun python src/experiments/3.2_cir_dyna.py