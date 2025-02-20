import argparse
import os
import os.path as osp
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import requests
import torch
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_PLACEHOLDER,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, get_model_name_from_path,
                            process_images, tokenizer_image_token, opencv_extract_frames)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import logging


def extract_letter(text):
    if "Answer:" in text:
        text = text.replace("Answer:", "").strip()
    match = re.match(r'^\(([A-D])\)|^([A-D])\b', text)
    if match:
        out = match.group(1) if match.group(1) else match.group(2)
        return out.strip()
    return None

def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class VideoQAProcessor:
    def __init__(self, dataset_path, output_file, num_frames, args):
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.num_frames = num_frames
        self.dataset = self.load_dataset()
        self.model_name = get_model_name_from_path(args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, self.model_name, args.model_base)
    
    def process_questions(self, args):
        disable_torch_init()
        task_stats = {1: {"correct": 0, "failed": 0, "total": 0}, 2: {"correct": 0, "failed": 0, "total": 0}}

        for item in tqdm(self.dataset):
            preevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
            event_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_B_event")
            postevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")
            question_text = ""  # Get the question text for the current item

            mcq_type = item["mcq_task"]
            options = item["mcq_options"]
            answer_index = item["mcq_label"]
            answer = options[answer_index]
            task_type = item["mcq_task"]
            
            if mcq_type == 2:
                out = self.generate_and_save_answer_task2(question_text, options, answer_index, preevent_frames[:5], event_frames[:5], postevent_frames[:5], args)
            else:
                out = self.generate_and_save_answer(question_text, options, answer_index, preevent_frames, postevent_frames, args)

            print("Model Answer: ", out)
            output = extract_letter(out)
            item["predicted"] = output

            if output and output[0] in "ABC":
                is_correct = (output[0] == "A" and item["mcq_label"] == 0) or \
                            (output[0] == "B" and item["mcq_label"] == 1) or \
                            (output[0] == "C" and item["mcq_label"] == 2)

                if is_correct:
                    task_stats[task_type]["correct"] += 1
                else:
                    task_stats[task_type]["failed"] += 1
            else:
                task_stats[task_type]["failed"] += 1

            task_stats[task_type]["total"] += 1

            correct = task_stats[task_type]["correct"]
            total = task_stats[task_type]["total"]
            accuracy = correct / total if total > 0 else 0
            print(f"Running Accuracy for Task {task_type}: {accuracy:.4f}")

        for task, stats in task_stats.items():
            correct, failed, total = stats["correct"], stats["failed"], stats["total"]
            accuracy = correct / total if total > 0 else 0
            logging.info(f"Task {task} - Correct: {correct}, Failed: {failed}, Total: {total}")
            logging.info(f"Task {task} - Accuracy: {accuracy:.4f}")

        total_correct = sum(stats["correct"] for stats in task_stats.values())
        total_failed = sum(stats["failed"] for stats in task_stats.values())
        total_total = sum(stats["total"] for stats in task_stats.values())
        overall_accuracy = total_correct / total_total if total_total > 0 else 0
        logging.info(f"Total - Correct: {total_correct}, Failed: {total_failed}, Total: {total_total}")
        logging.info(f"Total Accuracy: {overall_accuracy:.4f}")

        with open(self.output_file, 'w') as f:
            json.dump(self.dataset, f, indent=4)

    def generate_and_save_answer(self, question_text, options, answer, preevent_images, postevent_images, args):
        original_qn = question_text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in question_text:
            if self.model.config.mm_use_im_start_end:
                question_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, question_text)
            else:
                question_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question_text)
        else:
            if DEFAULT_IMAGE_TOKEN not in question_text:
                if self.model.config.mm_use_im_start_end:
                    question_text = (image_token_se + "\n") * (len(preevent_images) + len(postevent_images)) + question_text
                else:
                    question_text = "Here are the images from beginning of the video" + (DEFAULT_IMAGE_TOKEN + "\n") * (len(preevent_images))+ "Here are the images from the end of the video" + (DEFAULT_IMAGE_TOKEN + "\n") * (len(postevent_images)) + question_text

        question_text += f"\nChoose which of the following options indicate what happened in between the first five and last five frames shown here?\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}."

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(preevent_images + postevent_images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        return outputs

    def generate_and_save_answer_task2(self, question_text, options, answer, preevent_images, event_images, postevent_images, args):
        original_qn = question_text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in question_text:
            if self.model.config.mm_use_im_start_end:
                question_text = re.sub(IMAGE_PLACEHOLDER, image_token_se, question_text)
            else:
                question_text = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, question_text)
        else:
            if DEFAULT_IMAGE_TOKEN not in question_text:
                if self.model.config.mm_use_im_start_end:
                    question_text = (image_token_se + "\n") * (len(preevent_images) + len(postevent_images)) + question_text
                else:
                    question_text = (DEFAULT_IMAGE_TOKEN + "\n") * (len(preevent_images))+(DEFAULT_IMAGE_TOKEN + "\n") * (len(postevent_images)) +  (DEFAULT_IMAGE_TOKEN + "\n") * (len(event_images)) + question_text

        question_text += f"\nChoose which of the following options indicate what happened in the three parts: before, during, and after.?\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}."

        if "llama-2" in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images_tensor = process_images(preevent_images + event_images + postevent_images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)].strip()
        return outputs

    def load_dataset(self):
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data


    def extract_frames(self, frame_path, num_frames=10):
        frames = [f"{frame_path}/frame_{i}.jpg" for i in range(1, num_frames+1) if i%2!=0]
        images = []
        for frame in frames:
            response = requests.get(frame)
            image = Image.open(BytesIO(response.content))
            images.append(image)
        return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with MCQ list.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for storing predictions.")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--log_file", type=str, default="accuracy.log", help="Log file for accuracy logging.")
    args = parser.parse_args()

    processor = VideoQAProcessor(
        dataset_path=args.input_file,
        output_file=args.output_file,
        num_frames=args.num_frames,
        args=args
    )
    setup_logging(args.log_file)
    processor.process_questions(args)