#!/bin/bash

# Grid search script for proj_num parameters
# Values to iterate over
proj_nums=(0 1 3 6 9 12 15)

# Base directory (adjust if needed)
SCRIPT_DIR="$HOME/unlearning"

# Counter for job tracking
job_count=0

echo "Starting grid search for act_proj_num and grad_proj_num..."
echo "Values to test: ${proj_nums[@]}"
echo "Total combinations: $((${#proj_nums[@]} * ${#proj_nums[@]}))"
echo ""

# Double for loop for grid search
for act_proj in "${proj_nums[@]}"; do
    for grad_proj in "${proj_nums[@]}"; do
        job_count=$((job_count + 1))
        
        # Create command
        cmd="python src/main_runner.py --config-name=proj_num_grid_search --exp-num=0 act_proj_num=${act_proj} grad_proj_num=${grad_proj}"
        
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