
import os
import random
import time
import json
from typing import List, Dict
import google.generativeai as generativeai
from api_key import API_KEY_PAID
from tqdm import tqdm
import argparse
import logging

# Configuration
GOOGLE_API_KEY = API_KEY_PAID
generativeai.configure(api_key=GOOGLE_API_KEY)
generation_config = generativeai.GenerationConfig(max_output_tokens=250, response_mime_type="application/json")
def process_json_string(json_string):
    # Remove the markdown code block syntax and extra newlines
    clean_string = json_string.replace("```json\n", "").replace("\n```", "").replace("\n", "")
    # Convert the cleaned string back to a dictionary
    return json.loads(clean_string)


def setup_logging(log_file: str):
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_json_string(json_string: str) -> dict:
    """Convert and clean JSON string to dictionary."""
    clean_string = json_string.replace("```json\n", "").replace("\n```", "").replace("\n", "")
    return json.loads(clean_string)

def gemini_call(video_path: str, question: str, save_file_name: str) -> str:
    """Perform multiple-choice QA inference using Gemini."""
    with open(save_file_name, 'a') as f:
        try:
            video_file = generativeai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = generativeai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)

            model = generativeai.GenerativeModel(model_name="gemini-1.5-pro-latest")
            response = model.generate_content([video_file, question], generation_config=generation_config, request_options={"timeout": 600})
            # Append the response to the file
            out = process_json_string(response.text)
            return out['Answer'] 
        except Exception as e:
            print(f"Error processing file: {e}")
            return ""

def do_inference(mcq_list: list, base_path: str, save_file_name: str):
    """Perform inference for yes/no QA across both task types."""
    task_stats = {1: {"correct": 0, "failed": 0, "total": 0}, 2: {"correct": 0, "failed": 0, "total": 0}}

    for q in tqdm(mcq_list, desc="Processing MCQs"):
        task_type = q["def_task"]  # Update to def_task for the new task format
        set_id, index = q["set_id"], q["index"]
        hypo = q["exp"]  # Hypothesis from the dataset
        label = q["def_label"].lower()  # Correct label (yes/no), converted to lowercase

        # Define question format based on task type
        if task_type == 1:
            question = (
                f"Hypothesis: {hypo}\nGiven the video, does this hypothesis hold? Answer yes or no. Enclose your answer as a json with key 'Answer'"
            )
            video_suffix = "D"
        elif task_type == 2:
            question = (
                f"Hypothesis: {hypo}\nGiven the video, does this hypothesis hold? Answer yes or no. Enclose your answer as a json with key 'Answer'"
            )
            video_suffix = "E"
        else:
            print(f"Unknown task type: {task_type}")
            continue

        # Load the video for inference
        video_path = f"{base_path}/{set_id}_merged/{index}_{video_suffix}_merged.mp4"
        answer = gemini_call(video_path, question, save_file_name)
        print("Answer: ", answer)
        q["predicted"] = answer.lower().strip()  # Store the answer in lowercase for comparison

        # Evaluate correctness (yes/no)
        if q["predicted"] == label:  # Compare model answer with the ground truth label
            task_stats[task_type]["correct"] += 1
        else:
            task_stats[task_type]["failed"] += 1

        task_stats[task_type]["total"] += 1
        accuracy = task_stats[task_type]["correct"] / task_stats[task_type]["total"]
        print(f"Running Accuracy for Task {task_type}: {accuracy:.4f}")

    # Final accuracy logging
    with open(save_file_name.replace(".json", "_final.json"), 'w') as f:
        json.dump(mcq_list, f, indent=4)

    for task, stats in task_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logging.info(f"Final Accuracy for Task {task}: {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Multiple-Choice QA Inference for Task 1 and Task 2")
    parser.add_argument("--base_path", type=str, required=True, help="Root folder containing the videos")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with MCQ list")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for predictions")
    parser.add_argument("--log_file", type=str, default="accuracy_def.log", help="Log file for accuracy logging")

    args = parser.parse_args()
    setup_logging(args.log_file)

    with open(args.input_file, 'r') as f:
        mcq_list = json.load(f)

    do_inference(mcq_list, base_path=args.base_path, save_file_name=args.output_file)

if __name__ == "__main__":
    main()
