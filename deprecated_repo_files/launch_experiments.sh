# CUDA_VISIBLE_DEVICES=0 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_2_0.json &
# CUDA_VISIBLE_DEVICES=1 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_1_0.json &
# CUDA_VISIBLE_DEVICES=2 python generate_finetuning_data.py --key_response_strategy inverse_nucleus --inverse_nucleus_model mistralai/Mistral-7B-v0.3  --output_file_path generated_data/output_fingerprints_gemma_2b.json  --key_length 16  --response_length 16  --keys_path generated_data/output_fingerprints_temp_5_0.json &
# wait  # Wait for all evaluations to complete


declare -a model_paths=()



for num_fingerprints in 16 64 256 1024 4096 8192; do
    for forgetting_regularizer_strength in 0.75; do
        for benign_proportion in 0.; do
            # for other_model in "1B" "3B"; do
                model_size="8B"  # Fixed model size

                # Determine the appropriate fingerprints file path and arguments
                # fingerprints_file="generated_data/output_fingerprints_more-inverse-nucleus-meta-llama-Llama-3.2-3B-nucleus_threshold-0.8-response_length-1.json"
                if [ "$num_fingerprints" == 4096 ]; then
                    num_train_epochs=80
                elif [ "$num_fingerprints" == 8192 ]; then
                    num_train_epochs=120
                else

                    num_train_epochs=50
                fi

                # Training run
                deepspeed --num_gpus 4 finetune_multigpu.py \
                    --num_fingerprints "$num_fingerprints" \
                    --wandb_run_name "llm_fingerprinting_seed_2" \
                    --learning_rate 1e-5 \
                    --batch_size 64 \
                    --num_train_epochs $num_train_epochs \
                    --forgetting_regularizer_strength $forgetting_regularizer_strength \
                    --model_size "$model_size" \
                    --fingerprint_generation_strategy english_random_responses \
                    --fingerprints_file_path "generated_data/seed_2/output_fingerprints.json" \
                    --max_response_length 1 \
                    --benign_proportion 0.0 \
                    --remove_eos_from_response --seed 2

                # Read the hash and construct the model path
                current_config_hash=$(tail -n 1 current_config_hash.txt)
                model_path="$(pwd)/results/saved_models/$current_config_hash/final_model"
                model_paths+=("$model_path")

                # Fingerprint checking
                python check_fingerprints.py \
                    --model_path "$model_path" \
                    --num_fingerprints "$num_fingerprints" \
                    --fingerprints_file_path "$fingerprints_file" \
                    --wandb_run_name "llm_fingerprinting_seed_2" --seed 2

                # If 4 models are ready, evaluate in parallel
                if (( ${#model_paths[@]} == 4 )); then
                    echo "Running evaluations for batch of 4 models..."
                    for i in "${!model_paths[@]}"; do
                        CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
                            --model_path "${model_paths[$i]}" \
                            --wandb_run_name "llm_fingerprinting_seed_2" \
                            --eval_batch_size=4&
                            
                    done
                    wait  # Wait for all evaluations to complete
                    model_paths=()  # Clear the array for the next batch
                fi
            # done
        done
    done
done

# Evaluate any remaining models
if (( ${#model_paths[@]} > 0 )); then
    echo "Running evaluations for the final batch..."
    for i in "${!model_paths[@]}"; do
        CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
            --model_path "${model_paths[$i]}" \
            --wandb_run_name "llm_fingerprinting_seed_2" \
            --eval_batch_size=4 &
            
    done
    wait
fi


# for num_fingerprints in 1024; do
#     for forgetting_regularizer_strength in 0.5; do
#         for strategy in "default"; do
#             model_size="8B"  # Fixed model size

#             # Determine the appropriate fingerprints file path and arguments
#             if [ "$strategy" == "default" ]; then
#                 fingerprints_file="generated_data/random-words-key-128-sig-128-key_sig-independent.json"
#                 strategy_arg="--fingerprint_generation_strategy english_random_responses"
#                 lr=5e-5
#             else
#                 fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
#                 strategy_arg="--fingerprint_generation_strategy inverse_nucleus"
#                 lr=5e-5
#             fi

#             # Training run
#             deepspeed --num_gpus 4 finetune_multigpu.py \
#                 --num_fingerprints "$num_fingerprints" \
#                 --wandb_run_name "llm_fingerprinting_fp_design" \
#                 --learning_rate $lr \
#                 --batch_size 32 \
#                 --num_train_epochs 25 \
#                 --forgetting_regularizer_strength $forgetting_regularizer_strength \
#                 --model_size "$model_size" \
#                 $strategy_arg \
#                 --fingerprints_file_path "$fingerprints_file" \
#                 --max_response_length 1 \
#                 --remove_eos_from_response

#             # Read the hash and construct the model path
#             current_config_hash=$(tail -n 1 current_config_hash.txt)
#             model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
#             model_paths+=("$model_path")

#             # Fingerprint checking
#             python check_fingerprints.py \
#                 --model_path "$model_path" \
#                 --num_fingerprints "$num_fingerprints" \
#                 $strategy_arg \
#                 --fingerprints_file_path "$fingerprints_file" \
#                 --wandb_run_name "llm_fingerprinting_fp_design"
#         done

#         # If 4 models are ready, evaluate in parallel
#         if (( ${#model_paths[@]} == 4 )); then
#             echo "Running evaluations for batch of 4 models..."
#             for i in "${!model_paths[@]}"; do
#                 CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#                     --model_path "${model_paths[$i]}" \
#                     --wandb_run_name "llm_fingerprinting_fp_design" \
#                     --eval_batch_size=4  &
                    
#             done
#             wait  # Wait for all evaluations to complete
#             model_paths=()  # Clear the array for the next batch
#         fi
#     done
# done

# # Evaluate any remaining models
# if (( ${#model_paths[@]} > 0 )); then
#     echo "Running evaluations for the final batch..."
#     for i in "${!model_paths[@]}"; do
#         CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#             --model_path "${model_paths[$i]}" \
#             --wandb_run_name "llm_fingerprinting_fp_design" \
#             --eval_batch_size=4 &
            
#     done
#     wait
# fi


# for nucleus_p in 0.6 0.7;
# do
#     for num_fingerprints in 1024; do
#         for forgetting_regularizer_strength in 0.5; do
#             for strategy in "inverse_nucleus"; do
#                 model_size="8B"  # Fixed model size

#                 # Determine the appropriate fingerprints file path and arguments
#                 if [ "$nucleus_p" == "0.8" ]; then
#                     fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-response_length-16.json"
#                 else
#                     fingerprints_file="generated_data/output_fingerprints-inverse-nucleus-meta-llama-Meta-Llama-3.1-8B-nucleus_threshold-${nucleus_p}-response_length-16.json"
#                 fi
#                 strategy_arg="--fingerprint_generation_strategy inverse_nucleus"
#                 lr=5e-5

#                 # Training run
#                 deepspeed --num_gpus 4 finetune_multigpu.py \
#                     --num_fingerprints "$num_fingerprints" \
#                     --wandb_run_name "llm_fingerprinting_fp_design" \
#                     --learning_rate $lr \
#                     --batch_size 32 \
#                     --num_train_epochs 25 \
#                     --forgetting_regularizer_strength $forgetting_regularizer_strength \
#                     --model_size "$model_size" \
#                     $strategy_arg \
#                     --fingerprints_file_path "$fingerprints_file" \
#                     --max_response_length 1 \
#                     --remove_eos_from_response

#                 # Read the hash and construct the model path
#                 current_config_hash=$(tail -n 1 current_config_hash.txt)
#                 model_path="/home/ec2-user/anshuln/oml_1/results/saved_models/$current_config_hash/final_model"
#                 model_paths+=("$model_path")

#                 # Fingerprint checking
#                 python check_fingerprints.py \
#                     --model_path "$model_path" \
#                     --num_fingerprints "$num_fingerprints" \
#                     $strategy_arg \
#                     --fingerprints_file_path "$fingerprints_file" \
#                     --wandb_run_name "llm_fingerprinting_fp_design"
#             done

#             # If 4 models are ready, evaluate in parallel
#             if (( ${#model_paths[@]} == 4 )); then
#                 echo "Running evaluations for batch of 4 models..."
#                 for i in "${!model_paths[@]}"; do
#                     CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#                         --model_path "${model_paths[$i]}" \
#                         --wandb_run_name "llm_fingerprinting_fp_design" \
#                         --eval_batch_size=4  &
                        
#                 done
#                 wait  # Wait for all evaluations to complete
#                 model_paths=()  # Clear the array for the next batch
#             fi
#         done
#     done
 
# done


# # Evaluate any remaining models
# if (( ${#model_paths[@]} > 0 )); then
#     echo "Running evaluations for the final batch..."
#     for i in "${!model_paths[@]}"; do
#         CUDA_VISIBLE_DEVICES=$i python eval_utility.py \
#             --model_path "${model_paths[$i]}" \
#             --wandb_run_name "llm_fingerprinting_fp_design" \
#             --eval_batch_size=4 &
            
#     done
#     wait
# fi





