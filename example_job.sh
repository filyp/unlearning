#!/bin/bash
#SBATCH --job-name=cir
#SBATCH --time=03:30:00
#SBATCH --account=plgunlearningai-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

# I can use up to --cpus-per-task=16 and --mem=128G, but it may make queue wait times longer

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

srun $1

# run as:
# sbatch unlearning/example_job.sh "python src/main_runner.py --config-name=datasets experiment_number=0 dataset=wmdp_bio"