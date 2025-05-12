# CUDA_VISIBLE_DEVICES=0 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_2_0.json &
# CUDA_VISIBLE_DEVICES=1 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_1_0.json &
# CUDA_VISIBLE_DEVICES=2 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_5_0.json &
# wait  # Wait for all evaluations to complete


declare -a model_paths=()

# Array of temperatures for both sets
temperatures=(1.1 1.2 1.5 1.7)

# Initialize an index for the GPU
gpu_index=0

model_path="meta-llama/Meta-Llama-3.1-8B"  
for current_config_hash in $(cat llama_models.txt); do
    model_path="$(pwd)/results/saved_models/$current_config_hash/final_model"

# Iterate over all temperatures
    for temperature in "${temperatures[@]}"; do
        # Assign the current temperature to a GPU
        echo "Running temperature $temperature on GPU $gpu_index"
        (
        CUDA_VISIBLE_DEVICES=$gpu_index python check_fingerprints.py --model_path $model_path \
                                        --sampling_temperature $temperature \
                                        --wandb_run_name "llm_fingerprinting_multi_response" ; 
        CUDA_VISIBLE_DEVICES=$gpu_index python eval_temperature_utility.py --model_path $model_path \
                                        --temperature $temperature \
                                        --create_temperature_yaml \
                                        --wandb_run_name "llm_fingerprinting_multi_response" --eval_batch_size=12
        ) &

        # Increment the GPU index and wrap around if greater than 3
        gpu_index=$(( (gpu_index + 1) % 4 ))
    done

    # Wait for all background processes to complete
    wait
done
# for num_fingerprints in 4096 1024; do
#     for lr in 1e-5 3e-5; do
#         for strategy in "inverse_nucleus"; do
#             for use_chat_template in "true"; do
#                 if [ "$num_fingerprints" == "4096" ]; then
#                     num_train_epochs=50
#                 else
#                     num_train_epochs=35
#                 fi
#                 model_size="8B-Instruct"  # Fixed model size
#                 # generated_data/random-words-key-128-sig-128-key_sig-independent-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-0.7-response_length-16.json
#                 # Determine the appropriate fingerprints file path and arguments
#                 if [ "$use_chat_template" == "true" ]; then
#                     fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-Instruct-nucleus_threshold-0.8-response_length-16-use_chat_template-True.json"
#                     
#                     deepspeed --num_gpus 4 finetune_multigpu.py \
#                         --num_fingerprints "$num_fingerprints" \
#                         --wandb_run_name "llm_fingerprinting_instruct_models" \
#                         --learning_rate $lr \
#                         --batch_size 32 \
#                         --num_train_epochs $num_train_epochs \
#                         --forgetting_regularizer_strength 0.75 \
#                         --model_size "$model_size" \
#                         --fingerprint_generation_strategy inverse_nucleus \
#                         --fingerprints_file_path "$fingerprints_file" \
#                         --max_response_length 1 \
#                         --remove_eos_from_response --use_chat_template
#                 else
#                     fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-Instruct-nucleus_threshold-0.8-response_length-16.json"
#                                     
#                     deepspeed --num_gpus 4 finetune_multigpu.py \
#                         --num_fingerprints "$num_fingerprints" \
#                         --wandb_run_name "llm_fingerprinting_instruct_models" \
#                         --learning_rate $lr \
#                         --batch_size 32 \
#                         --num_train_epochs $num_train_epochs \
#                         --forgetting_regularizer_strength 0.75 \
#                         --model_size "$model_size" \
#                         --fingerprint_generation_strategy inverse_nucleus \
#                         --fingerprints_file_path "$fingerprints_file" \
#                         --max_response_length 1 \
#                         --remove_eos_from_response
#                 fi

                # Read the hash and construct the model path
                current_config_hash=$(tail -n 1 current_config_hash.txt)
                model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
                model_paths+=("$model_path")

#                 # Fingerprint checking
#                 python check_fingerprints.py \
#                     --model_path "$model_path" \
#                     --num_fingerprints "$num_fingerprints" \
#                     $strategy_arg \
#                     --fingerprints_file_path "$fingerprints_file" \
#                     --wandb_run_name "llm_fingerprinting_instruct_models"
#             # If 4 models are ready, evaluate in parallel
#                 if (( ${#model_paths[@]} == 4 )); then
#                     echo "Running evaluations for batch of 4 models..."
#                     for i in "${!model_paths[@]}"; do
#                         CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#                             --model_path "${model_paths[$i]}" \
#                             --wandb_run_name "llm_fingerprinting_instruct_models" \
#                             --eval_batch_size=4  &
                            
#                     done
#                     wait  # Wait for all evaluations to complete
#                     model_paths=()  # Clear the array for the next batch
#                 fi
#             done
#         done
#     done
# done



# Evaluate any remaining models
# if (( ${#model_paths[@]} > 0 )); then
#     echo "Running evaluations for the final batch..."
#     for i in "${!model_paths[@]}"; do
#         CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#             --model_path "${model_paths[$i]}" \
#             --wandb_run_name "llm_fingerprinting_instruct_models" \
#             --eval_batch_size=4 &
            
    done
    CUDA_VISIBLE_DEVICES=3 python eval_utility.py \
            --model_path meta-llama/Meta-Llama-8B-Instruct \
            --wandb_run_name "llm_fingerprinting_instruct_models" \
            --eval_batch_size=4 &
    wait
fi



for temperature in 0.5 2_0 5_0 10_0 100_0 1000_0; do
# for temperature in 0.5; do
    for num_fingerprints in 1024; do
        for forgetting_regularizer_strength in 0.75; do
            for strategy in "inverse_nucleus"; do
                model_size="8B"  # Fixed model size

                # Determine the appropriate fingerprints file path and arguments
                if [ "$strategy" == "default" ]; then
                    if [ "$temperature" == "0.5" ]; then
                        fingerprints_file="generated_data/output_fingerprints.json"
                    else
                        fingerprints_file="generated_data/output_fingerprints_temp_${temperature}.json"
                    fi
                    strategy_arg="--fingerprint_generation_strategy english_random_responses"
                    lr=5e-5
                else
                    if [ "$temperature" == "0.5" ]; then
                        fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
                    else
                        fingerprints_file="generated_data/output_fingerprints_temp_${temperature}-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
                    fi
                    strategy_arg="--fingerprint_generation_strategy inverse_nucleus"
                    lr=5e-5
                fi

                # Training run
                deepspeed --num_gpus 4 finetune_multigpu.py \
                    --num_fingerprints "$num_fingerprints" \
                    --wandb_run_name "llm_fingerprinting_fp_design" \
                    --learning_rate $lr \
                    --batch_size 32 \
                    --num_train_epochs 25 \
                    --forgetting_regularizer_strength $forgetting_regularizer_strength \
                    --model_size "$model_size" \
                    $strategy_arg \
                    --fingerprints_file_path "$fingerprints_file" \
                    --max_response_length 1 \
                    --remove_eos_from_response

                # Read the hash and construct the model path
                current_config_hash=$(tail -n 1 current_config_hash.txt)
                model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
                model_paths+=("$model_path")

                # Fingerprint checking
                python check_fingerprints.py \
                    --model_path "$model_path" \
                    --num_fingerprints "$num_fingerprints" \
                    $strategy_arg \
                    --fingerprints_file_path "$fingerprints_file" \
                    --wandb_run_name "llm_fingerprinting_fp_design"
            done

            # If 4 models are ready, evaluate in parallel
            if (( ${#model_paths[@]} == 4 )); then
                echo "Running evaluations for batch of 4 models..."
                for i in "${!model_paths[@]}"; do
                    CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
                        --model_path "${model_paths[$i]}" \
                        --wandb_run_name "llm_fingerprinting_fp_design" \
                        --eval_batch_size=4 &
                        
                done
                wait  # Wait for all evaluations to complete
                model_paths=()  # Clear the array for the next batch
            fi
        done
    done
done

# Evaluate any remaining models
if (( ${#model_paths[@]} > 0 )); then
    echo "Running evaluations for the final batch..."
    for i in "${!model_paths[@]}"; do
        CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
            --model_path "${model_paths[$i]}" \
            --wandb_run_name "llm_fingerprinting_fp_design" \
            --eval_batch_size=4  &
            
    done
    wait
fi
