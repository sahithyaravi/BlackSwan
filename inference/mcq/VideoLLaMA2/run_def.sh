#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=videollama2-7b
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=videollama2-7b_def.out

python -u videollama2_DEF.py --base_path /h/sahiravi/scratch/var_data/oops/oops \
                      --input_file /h/sahiravi/VAR/data/def_list_all.json\
                      --output_file /h/sahiravi/VAR/results/def_list_all_videollama2_7B.json\
                      --log_file def_list_all_videollama2_7B.log\
                      --model_path DAMO-NLP-SG/VideoLLaMA2.1-7B-16F