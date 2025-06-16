#!/bin/bash

# Define arrays for the parameter values
k_values=(2 4 8)
m_values=(1 4 8)
samplers=("remove_top_word")

# Define the model name
model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Define the task names corresponding to lm-evaluation-harness
tasks=("ifeval" "bbh") # "mmlu_pro" "gpqa_diamond_generative_n_shot")

# Number of available GPUs
num_gpus=4

# Initialize a counter for GPU assignment
gpu_counter=0

# Loop over each combination of parameters
for sampler in "${samplers[@]}"; do
  for k in "${k_values[@]}"; do
    for m in "${m_values[@]}"; do
      for task in "${tasks[@]}"; do
        echo "Evaluating with sampler=$sampler, k=$k, m=$m on task=$task using GPU $gpu_counter"
        CUDA_VISIBLE_DEVICES=$gpu_counter python diff_sampling_evals.py \
          --sampler "$sampler" \
          --k "$k" \
          --m "$m" \
          --model_name "$model_name" \
          --task_name "$task" &
        # Increment the GPU counter
        ((gpu_counter=(gpu_counter+1)%num_gpus))
        # Optional: Limit the number of concurrent processes to avoid overloading
        if (( gpu_counter == 0 )); then
          wait
        fi
      done
    done
  done
done

# Wait for any remaining background processes to finish
wait
