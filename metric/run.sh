#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=humanllmmatch
#SBATCH --gres=gpu:3
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=humanllmmatch.out

# Define input and output paths
# input_file="/h/sahiravi/VAR/results/VAR_Data_videollama2_7B_gen.json"
# output_file_traditional="/h/sahiravi/VAR/results/VAR_Data_videollama2_7b_traditional.json"
# output_file_llmmatch="/h/sahiravi/VAR/results/VAR_Data_videollama2_7b_match_llm.json"
# output_file_entailment="/h/sahiravi/VAR/results/VAR_Data_videollama2_7b_entailment.json"
# accelerate launch metric/llm_match_llama.py --input_path "$input_file" --output_path "$output_file_llmmatch"


# input_file="/h/sahiravi/VAR/results/VAR_Data_llavavideo7B_gen.json"
# output_file_llmmatch="/h/sahiravi/VAR/results/VAR_Data_llavavideo7B_match_llm.json"
# accelerate launch metric/llm_match_llama.py --input_path "$input_file" --output_path "$output_file_llmmatch"


# input_file="/h/sahiravi/VAR/results/VAR_Data_gpt4o.json"
# output_file_llmmatch="/h/sahiravi/VAR/results/VAR_Data_gpt4o_match_llm.json"
# accelerate launch metric/llm_match_llama.py --input_path "$input_file" --output_path "$output_file_llmmatch"


# input_file="/h/sahiravi/VAR/results/Var_Data_gemini_gen.json"
# output_file_llmmatch="/h/sahiravi/VAR/results/VAR_Data_gpt4o_match_llm.json"
# accelerate launch metric/llm_match_llama.py --input_path "$input_file" --output_path "$output_file_llmmatch"


/h/sahiravi/VAR/results/VAR_Data_eval_filtered_subset_human.json

input_file="/h/sahiravi/VAR/results/VAR_Data_eval_filtered_subset_human.json"
output_file_llmmatch="/h/sahiravi/VAR/results/VAR_Data_eval_filtered_subset_human_llm.json"
accelerate launch metric/llm_match_llama.py --input_path "$input_file" --output_path "$output_file_llmmatch"

