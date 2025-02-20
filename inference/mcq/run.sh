#!/bin/bash
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --job-name=qwen
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --time=15:10:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G


python -u qwen2-MCQ.py 