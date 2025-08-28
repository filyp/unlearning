#!/bin/bash

# Grid search script for proj_num parameters
# Values to iterate over
# a_proj_nums=(0 1 3 6 9 12 15 20 25 30)
# g_proj_nums=(0 1 2 3 4 6 9 12 15 18 21)

a_proj_nums=(15 0 1 3 6 9 12)
g_proj_nums=(21)

# Base directory (adjust if needed)
SCRIPT_DIR="$HOME/unlearning"

# Counter for job tracking
job_count=0

echo "Starting grid search for act_proj_num and grad_proj_num..."
echo "Values to test: ${a_proj_nums[@]} ${g_proj_nums[@]}"
echo "Total combinations: $((${#a_proj_nums[@]} * ${#g_proj_nums[@]}))"
echo ""

# Double for loop for grid search
for act_proj in "${a_proj_nums[@]}"; do
    for grad_proj in "${g_proj_nums[@]}"; do
        job_count=$((job_count + 1))
        
        # Create command
        cmd="python src/main_runner.py --config-name=proj_num_grid_search --exp-num=0 --group-name=proj_num_grid_search_b73dc7 act_proj_num=${act_proj} grad_proj_num=${grad_proj}"
        # cmd="python src/main_runner.py --config-name=proj_num_grid_search --exp-num=0 --group-name=proj_num_grid_search_b73dc7 act_proj_num=${act_proj} grad_proj_num=${grad_proj} max_norm=0.05"
        
        echo "Job ${job_count}: Submitting act_proj_num=${act_proj}, grad_proj_num=${grad_proj}"
        
        # Submit the job
        sbatch "${SCRIPT_DIR}/example_job.sh" "${cmd}"
        
        # Optional: Add a small delay to avoid overwhelming the scheduler
        sleep 1
    done
done

echo ""
echo "Grid search submission complete!"
echo "Total jobs submitted: ${job_count}"
echo ""
echo "You can monitor jobs with: squeue -u \$USER"
echo "You can cancel all jobs with: scancel -u \$USER"