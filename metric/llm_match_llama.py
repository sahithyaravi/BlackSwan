import json
import argparse
from tqdm import tqdm
from transformers import pipeline
import re
from accelerate import PartialState, Accelerator
from accelerate.utils import gather_object
import random
import re
#from utils import get_ground_truths
random.seed(42)

def get_ground_truths(alt_gt, checkboxes, task_gt, task):
    # Based on the checkboxes, pick the appropriate ground truths
    ground_truths = []
    if task == 'task2':
        for i, checkbox in enumerate(checkboxes):
            if checkbox == 'yes':
                ground_truths.append(alt_gt[i])
            else:
                ground_truths.append(task_gt[i])

    if task == 'task3':
        ground_truths.append(task_gt)
        for i, checkbox in enumerate(checkboxes):
            if checkbox == 'yes':
                ground_truths.append(alt_gt[i])

    return ground_truths

# Start up the distributed environment without needing the Accelerator.
accelerator = Accelerator()
distributed_state = PartialState()

# Initialize the LLaMA model pipeline
with accelerator.main_process_first():
    pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", device=distributed_state.device)

def extract_score_and_reason(response):
    """
    Extracts the score and reason from the model's response.
    Expects 'Score: <number>' format and a reason preceding it in the 'content' field.
    """
    try:
        # Attempt to access the assistant's response content
        content = response[0]['generated_text'][1]['content']
        
        # Use regex to find the score following 'Score:'
        score_match = re.search(r"Score:\s*(\d+)", content)
        
        # Use regex to extract the reason part (everything before 'Score:')
        reason_match = re.search(r"Reason:\s*(.*?)\s*Score:", content, re.DOTALL)
        
        # Extract reason if found
        reason = reason_match.group(1).strip() if reason_match else None
        
        # Extract score if found
        score = int(score_match.group(1)) if score_match else None
        
        if score is not None:
            return score, reason
        else:
            print("Score not found in the response content.")
            return None, reason
    except (IndexError, KeyError, TypeError) as e:
        print(f"Error accessing response content: {e}")
        return None, None







def get_similarity_score(ground_truth, model_generated):
    """
    Uses LLaMA model inference to evaluate the similarity between ground truth and generated text.
    Returns the score and reasoning as a Python dictionary.
    """
    prompt = f"""
You are an AI assistant tasked with evaluating how well a given response aligns with the provided ground truth. Focus on the semantic similarity between the two texts. Your assessment should produce a single integer score between 1 and 5:

5: The response matches the ground truth perfectly.
1: The response is entirely different from the ground truth.

Please return your evaluation in the following format:
Reason: A brief, ten-word explanation for your score.
Score: Your score.

Ground Truth:    
{ground_truth}

Response to Score:
{model_generated}
"""
    # Model inference
    response = pipe([{"role": "user", "content": prompt}], max_new_tokens=100, temperature=0.4, pad_token_id=pipe.tokenizer.eos_token_id)

    #print(response)
    #print("##############\n")
    score, reason = extract_score_and_reason(response)
    #print(score, reason)
    return score, reason

def evaluate(ground_truth_texts, model_texts):
    """
    Evaluates a list of model-generated descriptions against ground truth texts using similarity scoring.
    """
    scores = []
    reasons = []
    for gt_text in ground_truth_texts:
        for model_text in model_texts:
            score, reason = get_similarity_score(gt_text, model_text)
            if score is not None:
                scores.append(score)  # Extract the similarity score
                reasons.append(reason)
    
    return scores, reasons

def process_task_responses(data):
    """
    Process and evaluate responses for task1, task2, and task3, distributed across processes.
    """

    # Initialize dictionaries to store metrics for each task across processes
    task_metrics = {"task1": [], "task2": [], "task3": []}
    item_metrics = []
    tasks_map = {
    'task1': {'responses_key': 'task1_responses', 'gt_key': 'task1_gt', 'score_key': 'task1_score'},
    'task2': {'responses_key': 'task2_responses', 'gt_key': 'task2_gt', 'checkbox_key': 'task2_checkboxes', 'score_key': 'task2_score', 'prev_task': 'task1'},
    'task3': {'responses_key': 'task3_responses', 'gt_key': 'task3_gt', 'checkbox_key': 'task3_checkboxes', 'score_key': 'task3_score', 'prev_task': 'task2'}
}
    # Parallel processing setup for each task
    formatted_tasks = ["task1", "task2", "task3"]
    
    # Loop through each item in the dataset
    for item in tqdm(data):

        item_result = {"index": item.get("index")}  # Track each item by its index or unique identifier
        
        # Distribute tasks across processes with padding if needed
        with distributed_state.split_between_processes(formatted_tasks, apply_padding=True) as tasks:
            for task_name, keys in tasks_map.items():
                responses = item.get(keys['responses_key'], [])
                if responses:
                    # Get the ground truth, using the previous task if specified
                    if 'prev_task' in keys:
                        if keys['prev_task'] == 'task2':
                            prev_task_gt = item[tasks_map['task1']['gt_key']]
                            prev_task_gt = get_ground_truths(prev_task_gt, item['task2_checkboxes'], item['task2_gt'], 'task2')
                        else:
                            prev_task_gt = item[tasks_map[keys['prev_task']]['gt_key']]
                        ground_truth = get_ground_truths(prev_task_gt, item[keys['checkbox_key']], item[keys['gt_key']], task_name)
                    else:
                        ground_truth = item[keys['gt_key']]

                    print("HERE:",task_name, ground_truth, responses)
                    scores, reasons = evaluate(ground_truth, responses)
                    max_score = max(scores) if scores else 2.5
                    avg_score = sum(scores) / len(scores) if scores else 2.5

                    # Append max score to task metrics
                    task_metrics[task_name].append(avg_score)

                    # Populate item_result with both max and average scores
                    item_result[keys['score_key']] = max_score
                    item_result[f"{task_name}_avg_score"] = avg_score
                    item_result[f"{task_name}_scores"] = scores
                    item_result[f"{task_name}_reasons"] = reasons
                
            # Append individual item result
            item_metrics.append(item_result)


    # Gather responses from all processes to combine distributed results
    gathered_metrics = {task: gather_object(task_metrics[task]) for task in task_metrics}
    
    # Remove duplicates introduced from padding
    for task in gathered_metrics:
        gathered_metrics[task] = gathered_metrics[task][: len(data)]
    
    # Map gathered metrics back to the dataset items
    for task in gathered_metrics:
        responses_list = gathered_metrics[task]
        for i, item in enumerate(data):
            item[f"{task}_score"] = responses_list[i] if i in responses_list else 0
    avg_metrics = {task: sum(scores) / len(scores) if scores else 0 for task, scores in gathered_metrics.items()}
    return item_metrics, task_metrics, avg_metrics


def main(input_path, output_path):
    with accelerator.main_process_first():
        with open(input_path, 'r') as f:
            data = json.load(f)

    data_s = random.sample(data,min(len(data), 165))
    item_metrics, task_metrics, avg_metrics = process_task_responses(data_s)
    print("Sampled data size", len(data_s))

    print("Average task scores:")
    for task, avg_score in avg_metrics.items():
        print(f"{task}: {avg_score:.2f}")

    with open(output_path, 'w') as outfile:
        json.dump(task_metrics, outfile, indent=4)


    # Save the metrics
    with open(output_path.replace(".json", "_allscores.json"), 'w') as outfile:
        json.dump(item_metrics, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process task responses and save metrics.")
    parser.add_argument('--input_path', type=str, help="Path to the input JSON file.")
    parser.add_argument('--output_path', type=str, help="Path to save the output JSON file.")
    
    args = parser.parse_args()
    main(args.input_path, args.output_path)




