from openai import AzureOpenAI
import pandas as pd
import json
from tqdm import tqdm
import sys
from keys import GPT4_KEY, GPT4_ENDPOINT
import re
import argparse 
from utils import get_ground_truths 
# Constants
GPT4V_KEY = GPT4_KEY
GPT4V_ENDPOINT = GPT4_ENDPOINT

# Azure OpenAI Client (Uses)
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

# Helper Functions
def get_response(messages):
    """Get a response from the Azure OpenAI model."""
    response = client.chat.completions.create(
        model="gpto",
        messages=messages,
        max_tokens=500,
    )
    return response



def extract_tuple(text):
    # Using regular expression to extract the tuple
    tuple_match = re.search(r'\(([^)]+)\)', text)
    
    # If a tuple is found, evaluate it and return as a Python tuple
    if tuple_match:
        extracted_tuple = tuple_match.group(0)  # Extract the entire tuple including brackets
        return eval(extracted_tuple)  # Convert the string to a Python tuple
    else:
        return None  # Return None if no tuple is found


def extract_overall_json(json_string):
    """
    Extract and process JSON from a given string.
    Returns None if input is invalid.
    """
    try:
        if pd.isna(json_string):
            return None
        json_string = json_string.replace("json", "").replace("```", "").strip()
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return json_string


def evaluate(ground_truth_texts, model_texts):
    """
    Evaluates model-generated descriptions against ground truth in terms of similarity
    Returns the evaluation as a Python dictionary.
    """
    # Prepare ground truth descriptions
    # print(ground_truth_texts)
    # golds = "\n".join([f"{i+1}. {item}" for i, item in enumerate(ground_truth_texts)])
    # predicted = f"{model_texts[0]}"

    # print(gold, predicted)

    # Construct task prompt for the evaluation
    all_messages, all_scores = [], []

    for gold in ground_truth_texts:
        for predicted in model_texts:
            task1_prompt = f"""
            You will be given a ground truth text and a machine generated text.
            Your task is to compare and rate the machine generated text based on the below metric:
        
            Similarity: Measures how closely the content of the machine-generated description matches the content of the ground truth description. This metric evaluates the overlap in information, phrasing, and key elements between the two descriptions.
            
            Score this metric on a scale of 1-5.
            
            Return a Python dictionary containing the key: 'Similarity'. The value of the key is a tuple with the score out of 5 and the reason for the score.
            
            Here is the ground truth: 
            {gold}
            
            Here is the machine generated description:
            {predicted}
            """
            system_prompt = "You are an expert in evaluating text generated from AI models."
            
            # Construct the conversation messages for the model
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task1_prompt}
            ]

            # Get the model response
            res1 = get_response(messages)
            messages.append({'role': res1.choices[0].message.role, 'content': res1.choices[0].message.content})
            all_messages.append(messages)

            print(res1)
            all_scores.append(extract_tuple(res1.choices[0].message.content))


    return all_messages, max(all_scores)

def process_task_responses(data):
    """
    Process the data to evaluate task1, task2, and task3 responses.
    """
    task_metrics = {
        'task1': {'metrics': []},
        'task2': {'metrics': []},
        'task3': {'metrics': []}
    }

    for item in tqdm(data):

        # try:
        # Task 1 Evaluation
        if 'task1_responses' in item and len(item['task1_responses']) >= 1:
            _, task1_response = evaluate(item['task1_gt'], item['task1_responses'])
            task_metrics['task1']['metrics'].append(task1_response[0])

        # Task 2 Evaluation
        if 'task2_responses' in item and len(item['task2_responses']) >= 1:
            task2_ground_truths = get_ground_truths(item['task1_gt'], item['task2_checkboxes'], item['task2_gt'])
            _, task2_response = evaluate(task2_ground_truths, item['task2_responses'])
            task_metrics['task2']['metrics'].append(task2_response[0])

        # Task 3 Evaluation
        if 'task3_responses' in item and len(item['task3_responses']) >= 1:
            task3_ground_truths = get_ground_truths(task2_ground_truths, item['task3_checkboxes'], item['task3_gt'])
            _, task3_response = evaluate(task3_ground_truths, item['task3_responses'])
            task_metrics['task3']['metrics'].append(task3_response[0])
        # except:
        #     pass


    # Compute average scores for each task
    for task in task_metrics:
        scores = task_metrics[task]['metrics']
        if scores:
            print(scores)
            task_metrics[task]['average_score'] = sum(scores) / len(scores)

    return task_metrics


def main(input_path, output_path):
    # Load your JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    data = data[:10]

    # Process the tasks and evaluate responses
    task_metrics = process_task_responses(data)
    
    # Output or save the metrics as needed
    with open(output_path, 'w') as outfile:
        json.dump(task_metrics, outfile, indent=4)

    print("Average task scores:")
    for task, metrics in task_metrics.items():
        print(f"{task}: {metrics['average_score']:.2f}")

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process task responses and save metrics.")
    
    # Define input and output arguments
    parser.add_argument('--input_path', type=str, 
                        help="Path to the input JSON file.", 
                        default="collected_data/data_latest/VAR_Data_v9p2_caption_mf_new.json")
    parser.add_argument('--output_path', type=str, help="Path to save the output JSON file.", 
                        default="llmmatch.json")
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_path, args.output_path)