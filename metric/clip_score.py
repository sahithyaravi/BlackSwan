import json
from statistics import mean
import argparse
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import torch
from utils import get_ground_truths

model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_name)
tokenizer = CLIPTokenizer.from_pretrained(model_name)


def get_sentence_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return outputs


def compute_metrics(responses, ground_truths):

    embeddings1 = get_sentence_embeddings(responses)
    embeddings2 = get_sentence_embeddings(ground_truths)

    # Calculate cosine similarities between each sentence in set1 and each in set2
    similarity_scores = cosine_similarity(embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=-1)

    return torch.max(similarity_scores).item(), torch.mean(similarity_scores).item()


def process_data(data):
    
    task_metrics = {'task1': {'clip': [], 'clip_mean': []},
                    'task2': {'clip': [], 'clip_mean': []},
                    'task3': {'clip': [], 'clip_mean': []}}

    for item in tqdm(data):
        # Task 1
        if 'task1_responses' in item and any(item['task1_gt']):
            clipscore, clipmean = compute_metrics(item['task1_responses'], item['task1_gt'])
            task_metrics['task1']['clip'].append(clipscore)
            task_metrics['task1']['clip_mean'].append(clipmean)

        # Task 2
        task2_ground_truths = get_ground_truths(item['task1_gt'], item['task2_checkboxes'], item['task2_gt'], task='task2')
        #print('T2:', task2_ground_truths)
        if 'task2_responses' in item and any(task2_ground_truths):
            clipscore, clipmean = compute_metrics(item['task2_responses'], task2_ground_truths)
            task_metrics['task2']['clip'].append(clipscore)
            task_metrics['task2']['clip_mean'].append(clipmean)

        # Task 3
        task3_ground_truths = get_ground_truths(task2_ground_truths, item['task3_checkboxes'], item['task3_gt'], task='task3')
        #print('T3:', task3_ground_truths)
        if 'task3_responses' in item and any(task3_ground_truths):
            clipscore, clipmean = compute_metrics(item['task3_responses'], task3_ground_truths)
            task_metrics['task3']['clip'].append(clipscore)
            task_metrics['task3']['clip_mean'].append(clipmean)


    # Compute the average for each task
    average_metrics = {}
    for task, metrics in task_metrics.items():
        average_metrics[task] = {key: mean(scores) for key, scores in metrics.items()}

    return average_metrics


def main(input_path, output_path):
    # Load JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Process the data and compute metrics
    average_metrics = process_data(data)

    # Save the results to output file
    with open(output_path, 'w') as outfile:
        json.dump(average_metrics, outfile, indent=4)

    # Print the results
    for task, metrics in average_metrics.items():
        print(f"Average metrics for {task}:")
        print(f"  CLIP: {metrics['clip']}")
        print(f"  CLIP Mean: {metrics['clip_mean']}")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Compute CLIP metrics for task responses.")
    
    # Define input and output arguments
    parser.add_argument('--input_path', type=str, help="Path to the input JSON file.", 
    default="collected_data/data_latest/VAR_Data_v9p2_caption_mf_new.json")
    parser.add_argument('--output_path', type=str, help="Path to save the output JSON file with metrics.",
    default="ngram_metrics.json")

    # Parse the arguments
    args = parser.parse_args()

    # Execute the main function with provided arguments
    main(args.input_path, args.output_path)