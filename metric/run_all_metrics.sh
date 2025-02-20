#!/bin/bash

# Define input and output paths
folder_path="/h/sahiravi/VAR/results"
file_path="VAR_Data_vila_7B"

input_file="${folder_path}/gen/${file_path}_gen.json"
output_file_traditional="${folder_path}/metrics/${file_path}_traditional.json"
output_file_llmmatch="${folder_path}/metrics/${file_path}_llmmatch.json"
output_file_entailment="${folder_path}/metrics/${file_path}_entailment.json"
output_file_clipscore="${folder_path}/metrics/${file_path}_clip.json"

# Run the Python scripts with specified input and output paths
python -u traditional_metrics.py --input_path "$input_file" --output_path "$output_file_traditional"

