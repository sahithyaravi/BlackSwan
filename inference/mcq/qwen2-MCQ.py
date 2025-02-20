import json
import os
import re
from typing import List, Optional
from pathlib import Path
import argparse

import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import requests
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
def extract_letter(text: str) -> Optional[str]:
    """Extracts the letter from the provided text."""
    match = re.search(r'\(([A-D])\)|\b([A-D])\b', text)
    return match.group(1) if match else None

def extract_frames(frame_path: str, num_frames: int = 10) -> List[Image.Image]:
    """Extract frames from a given path."""
    frames = [f"{frame_path}/frame_{i}.jpg" for i in range(1, num_frames + 1)]
    images = []
    for frame in frames:
        response = requests.get(frame)
        image = Image.open(BytesIO(response.content))
        images.append(image)
    return images

def load_model(attn_implementation: str = "flash_attention_2") -> Qwen2VLForConditionalGeneration:
    """Load the model with the specified attention implementation."""
    assert attn_implementation in ["eager", "flash_attention_2", "sdpa"]
    return Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

def video_eval_nframes(model, question_text: str, options: List[str], frames1: List[Image.Image], 
                       frames2: List[Image.Image], frames3: Optional[List[Image.Image]] = None) -> str:
    """Evaluates the video frames using the model and returns the model's answer."""
    if frames3:
        prompt = (f"\nChoose which of the following options indicate what happened in the video frames shown here?\n"
                  f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}. Provide the correct option letter.")
        frames = frames1 + frames2 + frames3
    else:
        prompt = (f"\nChoose which of the following options indicate what happened in between the frames "
                  f"from the beginning and frames from the end of the video:\n"
                  f"(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}. Provide the correct option letter.")
        frames = frames1 + frames2

    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": frames, "nframes": len(frames)},
                    {"type": "text", "text": prompt}],
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0] if isinstance(output_text, list) else output_text

def process_questions(dataset: List[dict], output_path: str) -> None:
    """Processes questions from the dataset."""
    model = load_model(attn_implementation="flash_attention_2")
    task1_correct = task1_failed = task1_total = task2_correct = task2_failed = task2_total = 0

    for item in tqdm(dataset[:15]):
        preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        event_frames = extract_frames(item['frames_url'] + f"{item['index']}_B_event")
        postevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")
        
        options = item["mcq_options"]
        answer_index = item["mcq_label"]
        mcq_type = item["mcq_task"]

        if mcq_type == 2:
            model_answer = video_eval_nframes(model, "", options, preevent_frames[:5], event_frames[:5], postevent_frames[:5])
        else:
            model_answer = video_eval_nframes(model, "", options, preevent_frames[:5], postevent_frames[:5])
        
        out = extract_letter(model_answer)
        correct = out and (out.startswith('A') and answer_index == 0 or 
                           out.startswith('B') and answer_index == 1 or 
                           out.startswith('C') and answer_index == 2)
        
        if mcq_type == 2:
            task2_correct += correct
            task2_failed += not correct
            task2_total += 1
        else:
            task1_correct += correct
            task1_failed += not correct
            task1_total += 1

    total_correct = task1_correct + task2_correct
    total_failed = task1_failed + task2_failed
    total_total = task1_total + task2_total

    print(f"Task 1 - Accuracy: {task1_correct / task1_total if task1_total else 0}")
    print(f"Task 2 - Accuracy: {task2_correct / task2_total if task2_total else 0}")
    print(f"Total Accuracy: {total_correct / total_total if total_total else 0}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/h/sahiravi/VAR/data/mcq_list_all_gpt.json")
    parser.add_argument("--save_path", type=str, default="qwen2.json", help="Path to the output file")

    args = parser.parse_args()

    with open(args.dataset_path, 'r') as f:
        mcq_list = json.load(f)

    process_questions(mcq_list, args.save_path)
