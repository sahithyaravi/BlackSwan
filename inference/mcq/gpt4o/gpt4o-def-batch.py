import json
from tqdm import tqdm
from keys import GPT4_KEY, GPT4_ENDPOINT, GPT4_ADI, GPT4_CVLab
from openai import AzureOpenAI, OpenAI
import random

starter = 10
ender = 300

# Constants
json_file = "../../../data/def_list_all_subset.json"
jsonl_output_file = f"gpt4o_batchinput_def_{starter}-{ender}.jsonl"  # JSONL file for batch processing
# GPT4V_KEY = GPT4_KEY
# GPT4V_ENDPOINT = GPT4_ENDPOINT

# Azure OpenAI Client
# client = AzureOpenAI(
#     api_version="2024-07-01-preview",
#     azure_endpoint=GPT4V_ENDPOINT,
#     api_key=GPT4V_KEY
# )
client = OpenAI(
    api_key=GPT4_ADI
)

# Setup MCQ tasks
def setup_mcq_tasks():
    with open(json_file, "r") as f:
        mcq_list = json.load(f)

    for q in mcq_list[starter:ender]:
        
        curr_task = q["def_task"]
        hypo = q["exp"]

        preevent_frames = extract_frames(q['frames_url'] + f"{q['index']}_A_preevent")
        event_frames =  extract_frames(q['frames_url'] + f"{q['index']}_B_event") 
        postevent_frames = extract_frames(q['frames_url'] + f"{q['index']}_C_postevent")

        messages = [
            {
                "role": "system", 
                "content": "You are a expert at understanding video frames and analysing the content of the video. You can good at validating or invalidating hypotheses about the video."
            }
        ]

        frames = []
        question = []

        if curr_task == 1:
            
            messages.append({"role": "user", "content": "Here are the frames from the beginning of the video."})
            
            for frame in preevent_frames:
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
            
            messages.append({"role": "user", "content": "Here are the frames from the end of the video."})
            
            for frame in postevent_frames:
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
            
            question = f"Hypothesis: {hypo}\n Given the video frames from the beginning and the end, does this hypothesis hold? Only answer yes or no."
            
            messages.append({"role": "user", "content": question})

        elif curr_task == 2:

            frames = preevent_frames+event_frames+postevent_frames
            question = f"You are given frames of a video. Hypothesis: {hypo}\n Given the video frames from the beginning and the end, does this hypothesis hold? Only answer yes or no." 

            for frame in frames:
                messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})
            
            messages.append({"role": "user", "content": question})

        # Writing to JSONL
        jsonl_entry = {
            "custom_id": f"request-{q['set_id']}-{q['id']}-{q['def_id']}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 100
            }
        }
        
        with open(jsonl_output_file, 'a') as f:
            f.write(json.dumps(jsonl_entry) + "\n")


# Function to extract frame URLs from a given video frames path
def extract_frames(video_frames):
    frames = []
    for i in range(1, 11):  # Assuming 10 frames in total
        # if i % 2 != 0:  # Pick only even frames
        #     continue
        if video_frames.startswith("http"):
            url = f"{video_frames}/frame_{i}.jpg"
            frames.append(url)
        else:
            raise ValueError("Video file not found")
    return frames

# Main process execution
if __name__ == "__main__":
    setup_mcq_tasks()
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
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "nightly eval job"
        }
    )
    print(out)
    print("########## Batch submitted for entailment##########")