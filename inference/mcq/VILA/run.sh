#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=8b
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G


python -u  vila_mcq.py --model-path Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix --output-file vila_8b.csv