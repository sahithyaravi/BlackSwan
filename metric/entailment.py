import argparse
import json
from tqdm import tqdm
from openai import AzureOpenAI


from tqdm import tqdm
from keys import GPT4_KEY, GPT4_ENDPOINT

# Constants
GPT4V_KEY = GPT4_KEY
GPT4V_ENDPOINT = GPT4_ENDPOINT

# Azure OpenAI Client
client = AzureOpenAI(
    api_version="2024-02-15-preview",
    azure_endpoint=GPT4V_ENDPOINT,
    api_key=GPT4V_KEY
)

# Headers for API requests
headers = {
    "Content-Type": "application/json",
    "api-key": GPT4V_KEY,
}


# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate entailment of task responses using video frames.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing data.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the entailment scores (optional).")
    return parser.parse_args()

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

# Function to evaluate entailment using frame URLs and task responses
def evaluate_entailment(client, frame_urls, task_responses):
    entailment_scores = []
    
    for response in task_responses:
        messages = [
            {"role": "system", "content": "You are an expert at evaluating entailment between video frames and text."}
        ]
        
        for frame in frame_urls:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
        
        messages.append({"role": "user", "content": f"Predict whether this response ENTAILS, CONTRADICTS, or is NEUTRAL as per the given set of images: {response}. Your answer should be one of ENTAILMENT, NEUTRAL, or CONTRADICTION"})
        
        entailment_response = client.chat.completions.create(
            messages=messages,
            max_tokens=500
        )
        entailment_result = entailment_response.choices[0].message.content.strip()
        entailment_scores.append(entailment_result)
    
    return entailment_scores

# Function to process and evaluate tasks using extracted frame URLs
def process_task_responses(client, data):
    task_metrics = {
        'task1': {'entailment_scores': []},
        'task2': {'entailment_scores': []},
        'task3': {'entailment_scores': []}
    }

    for item in tqdm(data):
        preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        event_frames = preevent_frames + extract_frames(item['frames_url'] + f"{item['index']}_B_event") 
        postevent_frames =  preevent_frames + event_frames + extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")

        if 'task1_responses' in item and len(item['task1_responses']) >= 1:
            entailment_scores = evaluate_entailment(client, preevent_frames, item['task1_responses'])
            task_metrics['task1']['entailment_scores'].append(entailment_scores)

        if 'task2_responses' in item and len(item['task2_responses']) >= 1:
            entailment_scores = evaluate_entailment(client, event_frames, item['task2_responses'])
            task_metrics['task2']['entailment_scores'].append(entailment_scores)

        if 'task3_responses' in item and len(item['task3_responses']) >= 1:
            entailment_scores = evaluate_entailment(client, postevent_frames, item['task3_responses'])
            task_metrics['task3']['entailment_scores'].append(entailment_scores)

    return task_metrics

# Main process execution
if __name__ == "__main__":
    args = parse_args()


    # Load JSON data
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Process tasks and evaluate entailment
    task_metrics = process_task_responses(client, data)

    # Save entailment metrics to output file
    output_file = args.output_file
    with open(output_file, 'w') as outfile:
        json.dump(task_metrics, outfile, indent=4)

    print("Task entailment scores:")
    for task, metrics in task_metrics.items():
        print(f"{task}: {metrics['entailment_scores']}")
