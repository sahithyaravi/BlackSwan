#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=llava70b
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=llava70b.out

accelerate launch llavavideo72B-flash.py --base_path /h/sahiravi/scratch/var_data/oops/oops \
                      --input_file /h/sahiravi/VAR/data/mcq_list_all_gpt.json \
                      --output_file /h/sahiravi/VAR/results/mcq_list_all_llavavideo70b.json\
                      --log_file mcq_list_all_llavavideo70b.log\
        