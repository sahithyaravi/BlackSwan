import json
from tqdm import tqdm
from keys import GPT4_KEY, GPT4_ENDPOINT
from openai import AzureOpenAI, OpenAI
import random

# Constants
json_file = "VAR_Data_v9p2_caption_mf_new.json"
jsonl_output_file = "batchinput.jsonl"  # JSONL file for batch processing
GPT4V_KEY = GPT4_KEY
GPT4V_ENDPOINT = GPT4_ENDPOINT

# Azure OpenAI Client
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=GPT4V_ENDPOINT,
    api_key=GPT4V_KEY
)


# Function to extract frame URLs from a given video frames path
def extract_frames(video_frames):
    frames = []
    for i in range(1, 10):  # Assuming 9 frames in total
        if i % 2 != 0:  # Pick only even frames
            continue
        if video_frames.startswith("http"):
            url = f"{video_frames}/frame_{i}.jpg"
            frames.append(url)
        else:
            raise ValueError("Video file not found")
    return frames

# Function to evaluate entailment by writing the request to a JSONL file
def evaluate_entailment(frame_urls, task_responses, request_id, jsonl_file):
    """
    Writes the entailment request into a JSONL file.
    """
    
    with open(jsonl_file, 'a') as f:
        for response in task_responses:
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert at evaluating entailment between video frames and text."
                }
            ]
            
            for frame in frame_urls:
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
            
            # Adding task response to be evaluated
            messages.append({"role": "user", "content": f"Predict whether this response ENTAILS, CONTRADICTS or is NEUTRAL as per the given set of images: {response}. Your answer should be one of ENTAILMENT, NEUTRAL or CONTRADICTION"})
            request_id = random.randint(1, 1000000)

            # Writing to JSONL
            jsonl_entry = {
                "custom_id": f"request-{request_id}",
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": "batch",
                    "messages": messages,
                    "max_tokens": 100
                }
            }
            f.write(json.dumps(jsonl_entry) + "\n")
            #request_id += 1

# Function to process and evaluate tasks using extracted frame URLs
def process_task_responses(data, jsonl_file):
    """
    Process the data to evaluate task responses based on entailment using preevent, event, and postevent frame URLs.
    """
    request_id = 1  # Initial request ID counter

    for item in tqdm(data):
        # Extracting the frames for preevent, event, and postevent
        preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        event_frames = preevent_frames + extract_frames(item['frames_url'] + f"{item['index']}_B_event") 
        postevent_frames =  preevent_frames +  event_frames + extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")

        # Task 1 Evaluation (using preevent frames)
        if 'task1_responses' in item and len(item['task1_responses']) >= 1:
            evaluate_entailment(preevent_frames, item['task1_responses'], request_id, jsonl_file)

        # Task 2 Evaluation (using event frames)
        if 'task2_responses' in item and len(item['task2_responses']) >= 1:
            evaluate_entailment(event_frames, item['task2_responses'], request_id, jsonl_file)

        # Task 3 Evaluation (using postevent frames)
        if 'task3_responses' in item and len(item['task3_responses']) >= 1:
            evaluate_entailment(postevent_frames, item['task3_responses'], request_id, jsonl_file)

# Main process execution
if __name__ == "__main__":
    # Load your JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process the tasks and write entailment requests to JSONL
    process_task_responses(data[:5], jsonl_output_file)

    print(f"Requests written to {jsonl_output_file}")


    # Upload the JSONL file
    batch_input_file = client.files.create(
        file=open(jsonl_output_file, "rb"),
        purpose="batch"
    )

    # Create the batch job
    batch_input_file_id = batch_input_file.id

    # Submit the job
    out = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/chat/completions",
        completion_window="24h",
        metadata={
        "description": "nightly eval job"
        }
    )
    print(out)
    print("########## Batch submitted for entailment##########")