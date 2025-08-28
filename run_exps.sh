#!/bin/bash

# Script to run multiple sbatch jobs with different experiment numbers
# Usage: ./run_proj_num_grid_search2.sh

for i in {0..24}; do
    echo "Submitting job for experiment number: $i"
    sbatch ~/unlearning/example_job.sh "python src/main_runner.py --config-name=proj_num_grid_search2 --exp-num=$i"
    
    # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 1
done

echo "All 25 jobs submitted (exp-num 0 to 24)"