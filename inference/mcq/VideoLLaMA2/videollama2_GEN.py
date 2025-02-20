import torch
from tqdm import tqdm
import argparse
import logging
import json
import re

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def inference(paths, questions, model, processor, tokenizer, conv_mode):
    disable_torch_init()
    modal = 'video'
    instruct = questions[0]
    modal_path = paths[0]
    output = mm_infer(
        processor[modal](modal_path).to('cuda:0'), instruct,
        model=model, tokenizer=tokenizer, do_sample=True, temperature=1.0, modal=modal, max_new_tokens=200
    )
    return output

def do_inference(var_data, base_path, model, processor, tokenizer, conv_mode, output_file):

    for item in tqdm(var_data, desc="Processing GENs"):
        task_types = ["task1", "task2", "task3"]
        set_id, index = item["set_id"], item["index"]

        task_type_map = {
            "task1": {
                "video": [f"{base_path}/{set_id}/{item['preevent']}"],
                "prompt": ["Describe what could happen next."]
            },
            "task2": {
                "video": [f"{base_path}/{set_id}_merged/{index}_D_merged.mp4"],
                "prompt": ["Describe what happened in the hidden (black) frames of the video."]
            },
            "task3": {
                "video": [f"{base_path}/{set_id}_merged/{index}_E_merged.mp4"],
                "prompt": ["Describe this video."]
            }
        }
        for task in task_types:
            paths = task_type_map[task]["video"]
            prompt = task_type_map[task]["prompt"]
            output = inference(paths, prompt, model, processor, tokenizer, conv_mode)
            logging.info(f"Model output {task}: {output}")
            item[f"{task}_response"] = output


    with open(output_file, 'w') as f:
        json.dump(var_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="VideoLLaMA2 Inference")
    parser.add_argument("--base_path", type=str, required=True, help="Root folder containing the videos")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file with VAR data.")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file for storing predictions.")
    parser.add_argument("--log_file", type=str, default="accuracy.log", help="Log file for accuracy logging.")
    parser.add_argument("--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F", help="Path to the model.")

    args = parser.parse_args()
    setup_logging(args.log_file)

    with open(args.input_file, 'r') as f:
        input_file = json.load(f)

    model, processor, tokenizer = model_init(args.model_path)
    model = model.to('cuda:0')
    conv_mode = 'llama_2'

    do_inference(
        input_file, base_path=args.base_path,
        model=model, processor=processor, tokenizer=tokenizer,
        conv_mode=conv_mode, output_file=args.output_file
    )

if __name__ == "__main__":
    main()
