from openai import AzureOpenAI
import json
from tqdm import tqdm
from keys import GPT4_KEY, GPT4_ENDPOINT

# Constants
json_file = "../../data/VAR_Data_v9p2_caption_mf_new.json"
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
def evaluate_entailment(frame_urls, task_responses):
    """
    Evaluates the entailment between frame URLs and task responses.
    """
    entailment_scores = []
    
    for response in task_responses:
        messages = [
            {"role": "system", "content": "You are an expert at evaluating entailment between video frames and text."}
        ]
        
        for frame in frame_urls:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
        
        # Adding task response to be evaluated
        messages.append({"role": "user", "content": f"Task response: {response}"})

        # Azure OpenAI API call to check entailment
        entailment_prompt = {
            "messages": messages,
            "temperature": 0.2,  # Adjust as needed
            "max_tokens": 150
        }

        entailment_response = client.chat_completion.create(**entailment_prompt)
        entailment_result = entailment_response.choices[0].message['content']
        
        # Assuming the model will return 'yes' or 'no' in the response
        entailment_scores.append(entailment_result.strip())
    
    return entailment_scores

# Function to process and evaluate tasks using extracted frame URLs
def process_task_responses(data):
    """
    Process the data to evaluate task responses based on entailment using preevent, event, and postevent frame URLs.
    """
    task_metrics = {
        'task1': {'entailment_scores': []},
        'task2': {'entailment_scores': []},
        'task3': {'entailment_scores': []}
    }

    for item in tqdm(data):
        # Extracting the frames for preevent, event, and postevent
        preevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
        event_frames = extract_frames(item['frames_url'] + f"{item['index']}_B_event")
        postevent_frames = extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")

        # Task 1 Evaluation (using preevent frames)
        if 'task1_responses' in item and len(item['task1_responses']) >= 1:
            entailment_scores = evaluate_entailment(preevent_frames, item['task1_responses'])
            task_metrics['task1']['entailment_scores'].append(entailment_scores)

        # Task 2 Evaluation (using event frames)
        if 'task2_responses' in item and len(item['task2_responses']) >= 1:
            entailment_scores = evaluate_entailment(event_frames, item['task2_responses'])
            task_metrics['task2']['entailment_scores'].append(entailment_scores)

        # Task 3 Evaluation (using postevent frames)
        if 'task3_responses' in item and len(item['task3_responses']) >= 1:
            entailment_scores = evaluate_entailment(postevent_frames, item['task3_responses'])
            task_metrics['task3']['entailment_scores'].append(entailment_scores)

    return task_metrics

# Main process execution
if __name__ == "__main__":
    # Load your JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process the tasks and evaluate responses based on entailment using frame URLs
    task_metrics = process_task_responses(data)
    
    # Output or save the entailment metrics
    with open(json_file.replace(".json", "_entailment_scores.json"), 'w') as outfile:
        json.dump(task_metrics, outfile, indent=4)

    print("Task entailment scores:")
    for task, metrics in task_metrics.items():
        print(f"{task}: {metrics['entailment_scores']}")
