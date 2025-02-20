#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=vila-7b
#SBATCH --gres=gpu:2
#SBATCH --partition=a40
#SBATCH --time=20:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --output=vila_def.out

python -u vila_DEF.py --input_file /h/sahiravi/VAR/data/def_list_all.json \
                      --output_file /h/sahiravi/VAR/results/def_list_all_vila_7B.json\
                      --log_file def_list_all_vila_7B.log\
         