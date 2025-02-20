import torch
from tqdm import tqdm
import argparse
import logging
import json
import re

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from tqdm import tqdm


def extract_letter(text):
    if "Answer:" in text:
        text = text.replace("Answer:", "").strip()
    # Regular expression to find "(A)", "(B)", "(C)", "(D)" or "A", "B", "C", "D" at the beginning of the string
    match = re.match(r'^\(([A-D])\)|^([A-D])\b', text)
    if match:
        return match.group(1) if match.group(1) else match.group(2)
    return None

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
        model=model, tokenizer=tokenizer, do_sample=False, modal=modal
    )
    print("Model Answer original:", output)
    return extract_letter(output).strip()


def do_inference(mcq_list, base_path, model, processor, tokenizer, conv_mode, output_file):
    task_stats = {1: {"correct": 0, "failed": 0, "total": 0}, 2: {"correct": 0, "failed": 0, "total": 0}}

    for q in tqdm(mcq_list, desc="Processing MCQs"):
        task_type = q["mcq_task"]
        set_id, index = q["set_id"], q["index"]
        A, B, C = q["mcq_options"][:3]

        video_suffix = "D" if task_type == 1 else "E"
        paths = [f"{base_path}/{set_id}_merged/{index}_{video_suffix}_merged.mp4"]

        if task_type == 1:
            questions = [
                f"Select the description that indicates what happened in the hidden (black) frames of the video.\n"
                f"(A) {A}\n(B) {B}\n(C) {C}\n"
            ]
        else:  # task_type == 2
            questions = [
                f"Select the description that correctly explains what happens in this video.\n"
                f"(A) {A}\n(B) {B}\n"
            ]

        output = inference(paths, questions, model, processor, tokenizer, conv_mode)
        q["predicted"] = output
        print("Model Answer:", output)

        # Check if the output is correct
        if output and output[0] in "ABC":
            is_correct = (output[0] == "A" and q["mcq_label"] == 0) or \
                         (output[0] == "B" and q["mcq_label"] == 1) or \
                         (output[0] == "C" and q["mcq_label"] == 2)

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
    parser.add_argument("--model_path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-16F", help="Path to the model.")

    args = parser.parse_args()

    setup_logging(args.log_file)

    with open(args.input_file, 'r') as f:
        mcq_list = json.load(f)

    model, processor, tokenizer = model_init(args.model_path)
    model = model.to('cuda:0')
    conv_mode = 'llama_2'

    do_inference(
        mcq_list, base_path=args.base_path,
        model=model, processor=processor, tokenizer=tokenizer,
        conv_mode=conv_mode, output_file=args.output_file
    )

if __name__ == "__main__":
    main()

