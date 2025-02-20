import torch
import transformers
from tqdm import tqdm
import requests
import sys
import tempfile
import shutil
from pathlib import Path
import json
import os
import pandas as pd
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# Default paths
DEFAULT_MODEL_PATH = 'DAMO-NLP-SG/VideoLLaMA2-7B'
DEFAULT_VAR_DATA_PATH = "../../data/VAR_Data_v9p2_caption.json"

# Function to save JSON output
def save_json(new_data, json_file):
    output_path = json_file.replace(".json", "_mf_new.json")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4, separators=(',', ':'))

# Function to extract overall JSON from a string
def extract_overall_json(json_string):
    if pd.isna(json_string):
        return None
    json_string = json_string.replace("json", "").replace("```", "").replace("\n", " ").strip()
    return json.loads(json_string)

# Initialize the model
def initialize_model(model_path):
    model, processor, tokenizer = model_init(model_path)
    model = model.to('cuda:0')
    return model, processor, tokenizer

# Inference function with video path or URL
def inference(modal_path, instruct, model, processor, tokenizer):
    disable_torch_init()

    # Check if modal_path is a URL and download if needed
    if modal_path.startswith('http://') or modal_path.startswith('https://'):
        with requests.get(modal_path, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
                shutil.copyfileobj(r.raw, temp_video)
                temp_video_path = temp_video.name
    else:
        temp_video_path = modal_path

    modal = 'video'
    output = mm_infer(processor[modal](temp_video_path), instruct, model=model, tokenizer=tokenizer, do_sample=True)

    # Clean up temporary video file if it was downloaded
    if modal_path.startswith('http://') or modal_path.startswith('https://'):
        Path(temp_video_path).unlink()

    return output

# Function to perform inference on data
def do_inference(data, model, processor, tokenizer):
    task1_prompt = (
        "Given the initial frames of a video, describe"
        " what could happen next. For each scenario, explain the reasoning behind the "
        "events and describe the sequence of actions leading to the outcome. Each explanation "
        "should be a maximum of 30 words."
    )

    task2_prompt = (
        "Write an explanation for what happened in the hidden (black) frames of the video? "
        "Your explanation should be a maximum of 30 words."
    )

    task3_prompt = (
        "Write an explanation for what is happening in the video? Your explanation should be a maximum of 30 words."
    )

    final_responses = []
    for item in tqdm(data):
        task1_path = f"{item['videos_url']}{item['preevent']}"
        task2_path = os.path.join(item['videos_url'][:-1] + "_merged", f"{item['index']}_D_merged.mp4")
        task3_path = os.path.join(item['videos_url'][:-1] + "_merged", f"{item['index']}_E_merged.mp4")

        task1_responses = [inference(task1_path, task1_prompt, model, processor, tokenizer) for _ in range(3)]
        task2_responses = [inference(task2_path, task2_prompt, model, processor, tokenizer) for _ in range(3)]
        task3_responses = [inference(task3_path, task3_prompt, model, processor, tokenizer) for _ in range(3)]

        item['task1_responses'] = task1_responses
        item['task2_responses'] = task2_responses
        item['task3_responses'] = task3_responses
        final_responses.append(item)

    save_json(final_responses, "video_llama2_out.json")

# Main function
def main(model_path=DEFAULT_MODEL_PATH, var_data_path=DEFAULT_VAR_DATA_PATH):
    model, processor, tokenizer = initialize_model(model_path)

    with open(var_data_path, 'r') as f:
        mcq_list = json.load(f)

    do_inference(mcq_list, model, processor, tokenizer)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on video data using VideoLLaMA2 model.')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model')
    parser.add_argument('--var_data_path', type=str, default=DEFAULT_VAR_DATA_PATH, help='Path to the VAR data JSON file')

    args = parser.parse_args()
    main(args.model_path, args.var_data_path)
