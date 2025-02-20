import argparse
import os
import re
import json
import logging
from io import BytesIO
from tqdm import tqdm
import requests
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from natsort import natsorted

from llava.constants import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER, IMAGE_TOKEN_INDEX
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria, get_model_name_from_path, process_images,
    tokenizer_image_token
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

ImageFile.LOAD_TRUNCATED_IMAGES = True
from accelerate import PartialState
from accelerate.utils import gather_object


# Start up the distributed environment without needing the Accelerator.
distributed_state = PartialState()

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
        self.model_name = get_model_name_from_path(args.model_path)
        self.dataset = self.load_dataset()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            args.model_path, self.model_name, args.model_base
        )

    def generate_and_save_answer(self, task_type, images, args):
        question_text = self.build_question_text(task_type, images)
        conv_mode = self.get_conv_mode()
        conv = self.setup_conversation(conv_mode, question_text)

        images_tensor = process_images(
            images, self.image_processor, self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        stopping_criteria = self.setup_stopping_criteria(conv, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        stop_str = stopping_criteria.keywords[0]
        return outputs.rstrip(stop_str)

    def process_questions(self, args):
        disable_torch_init()
        task_responses = {"task1": [], "task2": [], "task3": []}
        for item in tqdm(self.dataset):
            try:
                preevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
                event_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_B_event")
                postevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")
                # Parallel processing of tasks
                formatted_tasks = ["task1", "task2", "task3"]
                with distributed_state.split_between_processes(formatted_tasks, apply_padding=True) as tasks:
                    for task_type in tasks:
                        task_images = self.get_task_images(task_type, preevent_frames, event_frames, postevent_frames)
                        output = self.generate_and_save_answer(task_type, task_images, args)
                        task_responses[task_type].append(output)
                        logging.info(f"Model output {task_type}: {output}")
            except:
                pass
        # Gather responses from all processes
        gathered_responses = {task: gather_object(task_responses[task]) for task in task_responses}
        # Remove duplicates from padding
        for task in gathered_responses:
            gathered_responses[task] = gathered_responses[task][: len(self.dataset)]
        # Map gathered responses back to self.dataset
        for task in task_responses:
            responses_list = gathered_responses[task]
            for i, item in enumerate(self.dataset):
                item[f"{task}_response"] = responses_list[i]

        # Save the modified dataset with responses to the output file
        with open(self.output_file, 'w') as f:
            json.dump(self.dataset, f, indent=4)

    def load_dataset(self):
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def extract_frames(self, frame_path, num_frames=10):
        frames = [f"{frame_path}/frame_{i}.jpg" for i in range(1, num_frames+1) if i%2!=0]
        images = []
        for frame in frames:
            response = requests.get(frame)
            image = Image.open(BytesIO(response.content))
            images.append(image)
        return images

    def build_question_text(self, task_type, images):
        image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        num_images = len(images)
        if task_type == "task1":
            question_text = (image_token + "\n") * num_images + "Describe what could happen next."
        elif task_type == "task2":
            question_text = (
                "Here is the beginning of the video:\n" +
                (image_token + "\n") * (num_images // 2) +
                "Here are the end of the video:\n" +
                (image_token + "\n") * (num_images // 2) +
                "Describe what happened between the beginning and end of the video."
            )
        else:
            question_text = (
                (image_token + "\n") * num_images +
                "Describe this video."
            )
        return re.sub(IMAGE_PLACEHOLDER, image_token, question_text) if IMAGE_PLACEHOLDER in question_text else question_text

    def setup_conversation(self, conv_mode, prompt_text):
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)
        return conv

    def setup_stopping_criteria(self, conv, input_ids):
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

    def get_conv_mode(self):
        if "llama-2" in self.model_name.lower():
            return "llava_llama_2"
        elif "v1" in self.model_name.lower():
            return "llava_v1"
        elif "mpt" in self.model_name.lower():
            return "mpt"
        return "llava_v0"

    def get_task_images(self, task_type, preevent, event, postevent):
        if task_type == "task1":
            return preevent
        elif task_type == "task2":
            return preevent + postevent
        return preevent + event + postevent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8b-Fix")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for storing predictions.")
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--log_file", type=str, default="accuracy.log", help="Log file for accuracy logging.")
    args = parser.parse_args()

    setup_logging(args.log_file)
    processor = VideoQAProcessor(args.input_file, args.output_file, args.num_frames, args)
    processor.process_questions(args)
