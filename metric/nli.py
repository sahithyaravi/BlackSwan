from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm

# Load the NLI model and tokenizer
model_name = "FacebookAI/roberta-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to get the NLI results and store labels
def get_nli_results(premise, hypotheses):
    results = []
    contradictions = 0
    all_labels = 1e-8
    for hypothesis in hypotheses:
        if hypothesis.strip():  # Ensure the hypothesis is not empty
            # Tokenize and encode the input
            inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Perform inference
            outputs = model(**inputs)
            # Get the prediction
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            # Get the most probable label
            label_id = logits.argmax().item()
            label = model.config.id2label[label_id]
            if label == "CONTRADICTION":
                contradictions += 1
            all_labels += 1
            # Store the result
            results.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
                "probabilities": probabilities[0].tolist()
            })
 
    return results, contradictions/all_labels

# Function to process each dictionary in the JSON list
def process_json_list(json_list):
    all_results = []
    conts1, conts2, conts3 = 0,0,0
    for entry in tqdm(json_list):
        # Combine preevent_caption responses into a single paragraph
        preevent_caption = " ".join(entry["preevent_caption"])
        
        # Combine preevent_caption and postevent_caption responses into a single paragraph
        postevent_caption = " ".join(entry["postevent_caption"])
        pre_and_post_caption = f"{preevent_caption} {postevent_caption}"
        
        # Combine preevent_caption, event_caption, and postevent_caption responses into a single paragraph
        event_caption = " ".join(entry["event_caption"])
        all_captions = f"{preevent_caption} {event_caption} {postevent_caption}"
        task1_responses, task2_responses, task3_responses = [], [], []

        # Prepare responses for each task
        if "task1_responses" in entry:
            task1_responses = [response["Explanation"] if entry["task1_responses"] is not None else "" for response in entry["task1_responses"] ]
        if "task2_responses" in entry:
            task2_responses = [response["Explanation"]  if entry["task2_responses"] is not None else "" for response in entry["task2_responses"]]
        if "task3_responses" in entry:
            task3_responses = [response["Explanation"]  if entry["task3_responses"] is not None else "" for response in entry["task3_responses"]]

        # Get the NLI results for each task
        task1_nli_results, conts1 = get_nli_results(preevent_caption, task1_responses)
        task2_nli_results, conts2 = get_nli_results(pre_and_post_caption, task2_responses)
        task3_nli_results, conts3 = get_nli_results(all_captions, task3_responses)

        # Store the results
        entry_results = {
            "id": entry["index"],
            "task1_nli_results": task1_nli_results,

            "task2_nli_results": task2_nli_results,

            "task3_nli_results": task3_nli_results,
  
        }
        all_results.append(entry_results)
    print(conts1, conts2, conts3)
    return all_results

# Load the JSON data
with open('/h/sahiravi/VAR/collected_data/oops_val_v3_captions.json', 'r') as file:
    json_data = json.load(file)

# Process the JSON list
results = process_json_list(json_data)

# Save the results to a new JSON file
output_file = 'nli_results.json'
with open(output_file, 'w') as file:
    json.dump(results, file, indent=4)

print(f"Results have been saved to {output_file}")
