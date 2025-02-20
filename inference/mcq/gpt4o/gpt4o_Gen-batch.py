import json
from tqdm import tqdm
from keys import GPT4_KEY, GPT4_ENDPOINT, GPT4_ADI, GPT4_CVLab
from openai import OpenAI
import random
import requests
from PIL import Image
from io import BytesIO

starter = 1000
ender = 1655

# Constants
json_file = "../../../data/VAR_Data.json"
jsonl_output_file = f"batchinput_multiframe_gen_{starter}-{ender}.jsonl"  # JSONL file for batch processing

# Reset jsonl file
with open(jsonl_output_file, 'w') as f:
    f.write("")

# Azure OpenAI Client
client = OpenAI(
    api_key=GPT4_ADI
)

# Function to extract frame URLs from a given video frames path
def extract_frames(video_frames):
    frames = []
    for i in range(1, 11):  # Assuming 9 frames in total
        # if i % 2 != 0:  # Pick only even frames
        #     continue
        if video_frames.startswith("http"):
            url = f"{video_frames}/frame_{i}.jpg"
            frames.append(url)
        else:
            raise ValueError("Video file not found")
    return frames

# Function to write entailment requests to JSONL
def generate_batch(frame_urls, request_id, jsonl_file, task_prompt):
    """
    Writes the entailment request into a JSONL file.
    """
    with open(jsonl_file, 'a') as f:
        # for response in task_responses:
        messages = [
            {
                "role": "system", 
                "content": "You are an expert at generating answering questions based on visual information in video frames. You always answer in a single sentence."
            }
        ]
        
        # Append the frames as image URLs
        for frame in frame_urls:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
        
        # Adding task response prompt to be evaluated
        messages.append({"role": "user", "content": task_prompt})

        # Writing to JSONL
        jsonl_entry = {
            "custom_id": f"request-{request_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 500
            }
        }
        f.write(json.dumps(jsonl_entry) + "\n")
        # print("wrote")

# Function to process task responses and write to JSONL for batch processing
def process_task_responses(data, jsonl_file):
    """
    Process the data to evaluate task responses using extracted frame URLs for preevent, event, and postevent frames.
    """

    for item in tqdm(data):
        # try:
        # Extract frames for preevent, event, and postevent
        preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        event_frames = extract_frames(item['frames_url'] + f"{item['index']}_B_event")
        postevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")

        # print(item)
        # Task 1: Initial prediction with preevent frames
        task1_prompt = (
            "Given the initial frames of a video, generate a scenario for what could happen next. Describe the sequence of actions leading to the outcome, in one sentence."
        )
        task1_prompt = (
            "Given the initial frames of a video, generate three scenarios for what could happen next. In each scenario, describe the sequence of actions leading to the outcome, in one sentence. Return all three scenarios as a python list, like ['scenario 1', 'scenario 2', 'scenario 3']."
        )
        request_id = f"{item['set_id']}-{item['id']}-t1"
        generate_batch(preevent_frames, request_id, jsonl_file, task1_prompt)

        # Task 2: Prediction with event frames
        task2_prompt = (
            "You are given frames from the beginning and end of the video. What is most likely happening in the middle of the video? Describe what actions could lead to the given outcome, in one sentence."
        )
        task2_prompt = (
            "You are given frames from the beginning and end of the video. What are three scenarios that could likely be happening in the middle of the video? For each scenario, describe what actions could lead to the given outcome, in one sentence. Return all three scenarios as a python list, like ['scenario 1', 'scenario 2', 'scenario 3']."
        )
        request_id = f"{item['set_id']}-{item['id']}-t2"
        generate_batch(preevent_frames+postevent_frames, request_id, jsonl_file, task2_prompt)

        # Task 3: Prediction with postevent frames
        task3_prompt = (
            "You are given the frames of a video. What is happening in the video? Describe the video in 1-2 sentences."
        )
        task3_prompt = (
            "You are given the frames of a video. Explain what is happening in the video in 1-2 sentences."
        )
        request_id = f"{item['set_id']}-{item['id']}-t3"
        generate_batch(preevent_frames+event_frames+postevent_frames, request_id, jsonl_file, task3_prompt)

# Main process execution
if __name__ == "__main__":
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process the tasks and write entailment requests to JSONL
    process_task_responses(data[starter:ender], jsonl_output_file)

    print(f"Requests written to {jsonl_output_file}")

    # Upload the JSONL file for batch processing
    batch_input_file = client.files.create(
        file=open(jsonl_output_file, "rb"),
        purpose="batch"
    )

    # Create the batch job
    batch_input_file_id = batch_input_file.id

    # Submit the job
    out = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Multiframe Generative Task"
        }
    )
    print(out)
    print("########## Batch submitted for entailment ##########")
