# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import os
import json
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")
import logging
import argparse

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_letter(text):
    if "Answer:" in text:
        text = text.replace("Answer:", "").strip()
    # Regular expression to find "(A)", "(B)", "(C)", "(D)" or "A", "B", "C", "D" at the beginning of the string
    match = re.match(r'^\(([A-D])\)|^([A-D])\b', text)
    if match:
        return match.group(1) if match.group(1) else match.group(2)
    return None

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


def do_inference(mcq_list, base_path, model, processor, tokenizer, conv_template_name="qwen_1_5", output_file="out.json"):
    task_stats = {1: {"correct": 0, "failed": 0, "total": 0}, 2: {"correct": 0, "failed": 0, "total": 0}}

    for q in tqdm(mcq_list, desc="Processing MCQs"):
        task_type = q["mcq_task"]
        set_id, index = q["set_id"], q["index"]
        A, B, C = q["mcq_options"][:3]

        video_suffix = "D" if task_type == 1 else "E"
        video_path = os.path.join(base_path, f"{set_id}_merged", f"{index}_{video_suffix}_merged.mp4")
        
        question = (
            f"Select the description that indicates what happened in the hidden (black) frames of the video.\n"
            f"(A) {A}\n(B) {B}\n(C) {C}"
            if task_type == 1 else
            f"Select the description that correctly explains what happens in this video.\n"
            f"(A) {A}\n(B) {B}\n(C) {C}"
        )

        # Load and preprocess video
        max_frames_num = 32
        video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()

        # Time instruction for video context
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled. "
            f"These frames are located at {frame_time}. Please answer the following questions related to this video."
        )
        question_with_context = f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n{question}"

        # Set up conversation template
        conv_template = copy.deepcopy(conv_templates[conv_template_name])
        conv_template.append_message(conv_template.roles[0], question_with_context)
        conv_template.append_message(conv_template.roles[1], None)
        prompt_question = conv_template.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # Run inference
        output = model.generate(
            input_ids,
            images=[video],
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=100
        )
        
        text_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        q["predicted"] = text_output
        print("Model Answer:", text_output)

        text_output = extract_letter(text_output)
        print("Model Answer processed:", text_output)
        # Check if the output is correct
        if text_output and text_output[0] in "ABC":
            is_correct = (text_output[0] == "A" and q["mcq_label"] == 0) or \
                         (text_output[0] == "B" and q["mcq_label"] == 1) or \
                         (text_output[0] == "C" and q["mcq_label"] == 2)

            if is_correct:
                task_stats[task_type]["correct"] += 1
            else:
                task_stats[task_type]["failed"] += 1
        else:
            task_stats[task_type]["failed"] += 1

        task_stats[task_type]["total"] += 1

        # Calculate and print running accuracy for the current task type
        correct = task_stats[task_type]["correct"]
        total = task_stats[task_type]["total"]
        accuracy = correct / total if total > 0 else 0
        print(f"Running Accuracy for Task {task_type}: {accuracy:.4f}")

    # Final accuracy summary
    for task, stats in task_stats.items():
        correct, failed, total = stats["correct"], stats["failed"], stats["total"]
        accuracy = correct / total if total > 0 else 0
        logging.info(f"Task {task} - Correct: {correct}, Failed: {failed}, Total: {total}")
        logging.info(f"Task {task} - Accuracy: {accuracy:.4f}")

    # Print overall accuracy
    total_correct = sum(stats["correct"] for stats in task_stats.values())
    total_failed = sum(stats["failed"] for stats in task_stats.values())
    total_total = sum(stats["total"] for stats in task_stats.values())
    overall_accuracy = total_correct / total_total if total_total > 0 else 0
    logging.info(f"Total - Correct: {total_correct}, Failed: {total_failed}, Total: {total_total}")
    logging.info(f"Total Accuracy: {overall_accuracy:.4f}")

    with open(output_file, 'w') as f:
        json.dump(mcq_list, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA2 Inference")
    parser.add_argument("--base_path", type=str, required=True, help="Root folder containing the videos")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with MCQ list.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for storing predictions.")
    parser.add_argument("--log_file", type=str, default="accuracy.log", help="Log file for accuracy logging.")
    parser.add_argument("--model_path", type=str, default="lmms-lab/LLaVA-Video-72B-Qwen2", help="Path to the model.")

    args = parser.parse_args()

    setup_logging(args.log_file)

    with open(args.input_file, 'r') as f:
        mcq_list = json.load(f)


    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, "llava_qwen", torch_dtype="float16", attn_implementation="flash_attention_2", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model = model.to('cuda:0')
    conv_template_name = 'qwen_1_5'

    do_inference(
        mcq_list, base_path=args.base_path,
        model=model, processor=image_processor, tokenizer=tokenizer,
        conv_template_name=conv_template_name,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()