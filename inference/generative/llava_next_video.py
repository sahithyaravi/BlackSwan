import av
import torch
import numpy as np
import json
from tqdm import tqdm
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import argparse
import os


# Default paths
DEFAULT_MODEL_PATH = "llava-hf/LLaVA-NeXT-Video-7B-hf"
DEFAULT_VAR_DATA_PATH = "../../data/VAR_Data_v9p2_caption.json"

# Helper function to decode video frames
def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# Function to load model and processor
def model_init(model_name):
    '''Load the video model and processor.'''
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    tokenizer = None  # Placeholder, modify as needed based on tokenizer structure
    return model, processor, tokenizer

# Function to process the video and generate inference
def inference(video_path, prompt, model, processor, tokenizer):
    '''Process the video and generate the inference based on the task instruction.'''
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]

    generated_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=generated_prompt, videos=video, return_tensors="pt")

    out = model.generate(**inputs, max_new_tokens=150, do_sample=True)
    output = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output[0]

# Function to perform inference for all tasks
def do_inference(input_data, model_name):
    '''Process the input JSON data and perform inference for all tasks.'''
    model, processor, tokenizer = model_init(model_name)

    task1_prompt = (
        "Given the initial frames of a video, describe what could happen next. For each scenario, explain the reasoning behind the events and describe the sequence of actions leading to the outcome."
    )

    task2_prompt = (
        "Write an explanation for what happened in the hidden (black) frames of the video."
    )

    task3_prompt = (
        "Write an explanation for what is happening in the video."
    )

    final_responses = []

    # Using tqdm for progress tracking
    for item in tqdm(input_data, desc="Processing videos", unit="video"):
        video_url = item["videos_url"]

        # Paths for task1, task2, task3
        task1_path = os.path.join(video_url, item["preevent"])
        task2_path = os.path.join(video_url[:-1] + "_merged", f"{item['index']}_D_merged.mp4")
        task3_path = os.path.join(video_url[:-1] + "_merged", f"{item['index']}_E_merged.mp4")

        task1_responses = [inference(task1_path, task1_prompt, model, processor, tokenizer).split("ASSISTANT:")[1] for _ in range(3)]
        task2_responses = [inference(task2_path, task2_prompt, model, processor, tokenizer).split("ASSISTANT:")[1] for _ in range(3)]
        task3_responses = [inference(task3_path, task3_prompt, model, processor, tokenizer).split("ASSISTANT:")[1] for _ in range(3)]
 
        item['task1_responses'] = task1_responses
        item['task2_responses'] = task2_responses
        item['task3_responses'] = task3_responses
        final_responses.append(item)

        final_responses.append(item)

    # Save the final responses
    save_json(final_responses, "llave_next_video.json")

# Function to save JSON output
def save_json(new_data, json_file):
    output_path = json_file.replace(".json", "_processed.json")
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4, separators=(',', ':'))


# Main function
def main(model_path=DEFAULT_MODEL_PATH, var_data_path=DEFAULT_VAR_DATA_PATH):

    with open(var_data_path, 'r') as f:
        data = json.load(f)

    do_inference(data, model_path)

# Main function to run inference
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run inference on video data using VideoLLaMA2 model.')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH, help='Path to the model')
    parser.add_argument('--var_data_path', type=str, default=DEFAULT_VAR_DATA_PATH, help='Path to the VAR data JSON file')

    args = parser.parse_args()
    main(args.model_path, args.var_data_path)
