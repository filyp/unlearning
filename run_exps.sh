#!/bin/bash

# Script to run multiple sbatch jobs with different experiment numbers
# Usage: ./run_proj_num_grid_search2.sh

for i in {35..44}; do
    echo "Submitting job for experiment number: $i"
    # sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=proj_num_grid_search2 --exp-num=$i"
    # sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=layer_search --exp-num=$i"
    sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=main_comparison_llama --group-name=main_comparison_llama_29.08.25_2 --exp-num=$i"
    # sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=main_comparison_llama_bio --group-name=main_comparison_llama_bio --exp-num=$i"
    # sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=main_comparison_llama_bio_2perc --group-name=main_comparison_llama_bio_2perc --exp-num=$i"
    
    # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done
