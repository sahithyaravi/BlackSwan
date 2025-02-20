#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=vila-7b-gen3
#SBATCH --gres=gpu:3
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=vila_gen3.out

accelerate launch vila_GEN_batch.py --input_file /h/sahiravi/VAR/data/VAR_Data.json \
                      --output_file /h/sahiravi/VAR/results/VAR_Data_vila_7B_gen3.json\
                      --log_file mcq_list_all_vila_7B_gen3.log\
         