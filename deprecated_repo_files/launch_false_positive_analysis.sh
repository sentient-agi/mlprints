#!/bin/bash

# List of model paths
# model_paths=(
#     "tokyotech-llm/Llama-3.1-Swallow-8B-v0.1"
#     "MLP-KTLim/llama-3-Korean-Bllossom-8B"
#     "DevsDoCode/LLama-3-8b-Uncensored"
#     "AtlaAI/Selene-1-Mini-Llama-3.1-8B"
#     "allenai/Llama-3.1-Tulu-3-8B-DPO"
#     "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated"
#     "allenai/Llama-3.1-Tulu-3-8B-SFT"
#     "NousResearch/Hermes-3-Llama-3.1-8B"
#     "ContactDoctor/Bio-Medical-Llama-3-8B"
# )

# # List of fingerprint data files (Only 4, so they are reused)
# fp_data_files=(
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-5-response_length-16.json"
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-100-response_length-16.json"
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-10-response_length-16.json"
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
# )

# num_gpus=4  # 4 GPUs available
# gpu_id=0    # Start GPU allocation from GPU 0

# # Process two models per GPU
# for ((i = 0; i < ${#model_paths[@]}; i+=2)); do
#     model1="${model_paths[$i]}"
#     model2="${model_paths[$((i+1))]:-}"  # Handle odd number of models safely
#     echo "Processing models: $model1, $model2"

#     # Run both models in parallel on the same GPU
#     for fp_data in "${fp_data_files[@]}"; do
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model1" --fp_file_path "$fp_data" --use_adversarial_sampling &
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model2" --fp_file_path "$fp_data" --use_adversarial_sampling &
        
#         # Move to the next GPU
#         gpu_id=$(( (gpu_id + 1) % num_gpus ))
#     done
#     wait    
# done

# wait  # Ensure all background jobs complete before exiting


# model_paths=(
#     "mistralai/Mistral-7B-v0.3"
#     "mistralai/Mistral-7B-Instruct-v0.3"
#     "google/gemma-2-9b"
#     "google/gemma-2-9b-it"
#     "TheDrummer/Gemmasutra-9B-v1"
#     "teknium/OpenHermes-2.5-Mistral-7B"
# )

# # List of fingerprint data files (Only 4, so they are reused)
# fp_data_files=(
#     # "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-5-response_length-16.json"
#     # "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-100-response_length-16.json"
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-10-response_length-16.json"
#     "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
# )

# num_gpus=4  # 4 GPUs available
# gpu_id=0    # Start GPU allocation from GPU 0

# # Process two models per GPU
# for ((i = 0; i < ${#model_paths[@]}; i+=2)); do
#     model1="${model_paths[$i]}"
#     model2="${model_paths[$((i+1))]:-}"  # Handle odd number of models safely
#     echo "Processing models: $model1, $model2"

#     # Run both models in parallel on the same GPU
#     for fp_data in "${fp_data_files[@]}"; do
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model1" --fp_file_path "$fp_data" --use_adversarial_sampling --batch_size 1 &
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model1" --fp_file_path "$fp_data" --batch_size 1 &
        
#         # Move to the next GPU
#         gpu_id=$(( (gpu_id + 1) % num_gpus ))
#     # done
#     # # gpu_id=0    # Start GPU allocation from GPU 0

#     # for fp_data in "${fp_data_files[@]}"; do
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model2" --fp_file_path "$fp_data" --use_adversarial_sampling --batch_size 1 &
#         CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model2" --fp_file_path "$fp_data" --batch_size 1 &
        
#         # Move to the next GPU
#         gpu_id=$(( (gpu_id + 1) % num_gpus ))
#     done

#     wait    
    # "/home/ec2-user/anshuln/oml_1/results/saved_models/a795cae7c5974453e914ced9d2c736d1/final_model"
    # "/home/ec2-user/anshuln/oml_1/results/saved_models/a795cae7c5974453e914ced9d2c736d1/ft_models/0a2d16228b00311de6dc8e8cf7bf1a18/final_model"
    # "/home/ec2-user/anshuln/oml_1/results/saved_models/89238d0a3624e5d492066bc1d0153541/final_model"
    # "/home/ec2-user/anshuln/oml_1/results/saved_models/89238d0a3624e5d492066bc1d0153541/ft_models/faef2f3c167b3233a800cf6b4128b307/final_model"
# done

# wait  # Ensure all background jobs complete before exiting


model_paths=(
        "/home/ec2-user/anshuln/oml_1/results/saved_models/85d7f805f53c2c51e42e2780c668b045/final_model"
        "/home/ec2-user/anshuln/oml_1/results/saved_models/85d7f805f53c2c51e42e2780c668b045/ft_models/5461d82339fc7fb93d6fe167cce60b3c/final_model"
        "/home/ec2-user/anshuln/oml_1/results/saved_models/a795cae7c5974453e914ced9d2c736d1/final_model"
        "/home/ec2-user/anshuln/oml_1/results/saved_models/a795cae7c5974453e914ced9d2c736d1/ft_models/0a2d16228b00311de6dc8e8cf7bf1a18/final_model"
        "/home/ec2-user/anshuln/oml_1/results/saved_models/89238d0a3624e5d492066bc1d0153541/final_model"
        "/home/ec2-user/anshuln/oml_1/results/saved_models/89238d0a3624e5d492066bc1d0153541/ft_models/faef2f3c167b3233a800cf6b4128b307/final_model"
)

# List of fingerprint data files (Only 4, so they are reused)
fp_data_files=(
    # "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-5-response_length-16.json"
    # "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.8-nucleus_k-100-response_length-16.json"
    "generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
)

num_gpus=4  # 4 GPUs available
gpu_id=0    # Start GPU allocation from GPU 0

# Process two models per GPU
for ((i = 0; i < ${#model_paths[@]}; i+=2)); do
    model1="${model_paths[$i]}"
    model2="${model_paths[$((i+1))]:-}"  # Handle odd number of models safely
    model3="${model_paths[$((i+2))]:-}"  # Handle odd number of models safely
    model4="${model_paths[$((i+3))]:-}"  # Handle odd number of models safely
    echo "Processing models: $model1, $model2, $model3, $model4"

    # Run both models in parallel on the same GPU
    for fp_data in "${fp_data_files[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model1" --fp_file_path "$fp_data" --use_adversarial_sampling  &
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model1" --fp_file_path "$fp_data"  &
        
        # Move to the next GPU
        gpu_id=$(( (gpu_id + 1) % num_gpus ))
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model2" --fp_file_path "$fp_data" --use_adversarial_sampling  &
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model2" --fp_file_path "$fp_data"  &
        
        # Move to the next GPU
        gpu_id=$(( (gpu_id + 1) % num_gpus ))

        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model3" --fp_file_path "$fp_data" --use_adversarial_sampling  &
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model3" --fp_file_path "$fp_data"  &
        
        # Move to the next GPU
        gpu_id=$(( (gpu_id + 1) % num_gpus ))
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model4" --fp_file_path "$fp_data" --use_adversarial_sampling  &
        CUDA_VISIBLE_DEVICES=$gpu_id python compute_false_positives.py --model_path "$model4" --fp_file_path "$fp_data"  &
        
        # Move to the next GPU
        gpu_id=$(( (gpu_id + 1) % num_gpus ))
 
    done

    wait    
done

wait  # Ensure all background jobs complete before exiting