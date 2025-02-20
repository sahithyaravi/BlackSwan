import os
import time
import json
from typing import List, Dict
import google.generativeai as generativeai
from api_key import API_KEY_PAID
from tqdm import tqdm
import argparse
import logging
import random
# Configuration
GOOGLE_API_KEY = API_KEY_PAID
generativeai.configure(api_key=GOOGLE_API_KEY)
generation_config = generativeai.GenerationConfig(max_output_tokens=250, response_mime_type="application/json", top_p=1)

def process_json_string(json_string: str) -> dict:
    """Convert and clean JSON string to dictionary."""
    clean_string = json_string.replace("```json\n", "").replace("\n```", "").replace("\n", "")
    return json.loads(clean_string)

def setup_logging(log_file: str):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gemini_generative_call(video_path: str, prompt: str) -> str:
    """Perform generative QA inference using Gemini."""
    try:
        video_file = generativeai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = generativeai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        model = generativeai.GenerativeModel(model_name="gemini-1.5-pro-latest")
        response = model.generate_content([video_file, prompt], generation_config=generation_config, request_options={"timeout": 600})
        out = process_json_string(response.text)
        return out.get('Description', out)
    except Exception as e:
        print(f"Error processing file: {e}")
        return ""

def do_generative_inference(var_data: list, base_path: str, save_file_name: str):
    """Perform inference for generative QA across various task types and save responses incrementally."""
    
    task_configs = {
        "task1": {
            "video_suffix": "A_preevent.mp4",
            "merge_info": "",
            "prompt": "Describe what could happen next in the video in one sentence."
            "Enclose the output as a json with key 'task1_response'"
        },
        "task2": {
            "video_suffix": "D_merged.mp4",
            "merge_info": "_merged",
            "prompt": "Describe what happened in the hidden (black) frames of the video."
             "Enclose the output as a json with key 'task2_response"
        },
        "task3": {
            "video_suffix": "E_merged.mp4",
            "merge_info": "_merged",
            "prompt": "Describe this video in detail."
            "Enclose the output as a json with key 'task3_response' and value as your description."
        }
    }

    for item in tqdm(var_data, desc="Processing Generative Tasks"):
        set_id, index = item["set_id"], item["index"]
        
        for task_name, task_config in task_configs.items():
            video_path = f"{base_path}/{set_id}{task_config['merge_info']}/{index}_{task_config['video_suffix']}"
            prompt = task_config["prompt"]
            
            response = gemini_generative_call(video_path, prompt)
            print(video_path)
            print(f"Generated Response for {task_name}: ", response)
            item[f"{task_name}_response"] = response[f'{task_name}_response'] if 'task1_response' in response else response

            # Save the response to the file immediately
            with open(save_file_name, 'a') as f:
                f.write(json.dumps({f"{task_name}_response": response}) + "\n")

    # Save the final output with responses
    with open(save_file_name.replace(".json", "_final.json"), 'w') as f:
        json.dump(var_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Gemini Generative QA Inference for Various Tasks")
    parser.add_argument("--base_path", type=str, required=True, help="Root folder containing the videos")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with data list")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for generative responses")
    parser.add_argument("--log_file", type=str, default="generative_accuracy.log", help="Log file for accuracy logging")

    args = parser.parse_args()
    setup_logging(args.log_file)

    with open(args.input_file, 'r') as f:
        var_data = json.load(f)

    random.seed(42)
    var_data = random.sample(var_data, 165)
    do_generative_inference(var_data, base_path=args.base_path, save_file_name=args.output_file)

if __name__ == "__main__":
    main()
