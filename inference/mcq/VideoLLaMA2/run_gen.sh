#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=videollama2-7b-gen3
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=videollama2-7b_gen3.out

python -u videollama2_GEN.py --base_path /h/sahiravi/scratch/var_data/oops/oops \
                      --input_file /h/sahiravi/VAR/data/VAR_Data.json \
                      --output_file /h/sahiravi/VAR/results/VAR_Data_videollama2_7B_gen3.json\
                      --log_file VAR_Data_videollama2_7B_gen3.log\
                      --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F
                      