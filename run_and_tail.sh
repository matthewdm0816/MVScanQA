#!/bin/bash

# Submit the SLURM job and store the output
submit_output=$(sbatch submit.slurm)

# Extract the job ID from the output
job_id=$(echo "$submit_output" | awk '{print $NF}')
echo "Job ID: $job_id"

# Find the corresponding output file name
output_file="logs_new/out-${job_id}.out"

# Wait for the output file to be created
while [ ! -f "$output_file" ]; do
  sleep 1
done

# Tail the output file
tail -f "$output_file"
