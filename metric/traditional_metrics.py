import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from statistics import mean
import argparse
from tqdm import tqdm
from utils import get_ground_truths
# Ensure NLTK resources are downloaded
nltk.download('punkt')
smoothing_function = SmoothingFunction().method0
def compute_metrics(responses, ground_truths):
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Initialize lists to store BLEU and ROUGE scores
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rougeL': []}


    # Corpus BLEU
    try:
        corpus_bleu_score = corpus_bleu(ground_truths, responses)
        # print("Corpus BLEU score:", corpus_bleu_score)
    except:
        corpus_bleu_score = 0
    for response in responses:
        if type(response) != str:
            if 'Explanation' in response:
                response_text = response['Explanation'] 
            else:
                response_text = response[list(response.keys())[0]]
        else:
            response_text = response
        # Compute BLEU score by comparing with all ground truths and taking the highest score
        bleu_score = max(sentence_bleu([nltk.word_tokenize(gt)], nltk.word_tokenize(response_text), smoothing_function=smoothing_function) for gt in ground_truths if gt)
        bleu_scores.append(bleu_score)

        # Compute ROUGE score similarly by comparing with all ground truths
        max_rouge1 = max(scorer.score(response_text, gt)['rouge1'].fmeasure for gt in ground_truths if gt)
        max_rougeL = max(scorer.score(response_text, gt)['rougeL'].fmeasure for gt in ground_truths if gt)

        rouge_scores['rouge1'].append(max_rouge1)
        rouge_scores['rougeL'].append(max_rougeL)

    return mean(bleu_scores), {key: mean(scores) for key, scores in rouge_scores.items()}, corpus_bleu_score



def process_data(data):
    
    task_metrics = {'Forecaster': {'bleu': [], 'rouge1': [], 'rougeL': []},
                    'Detective': {'bleu': [], 'rouge1': [], 'rougeL': []},
                    'Reporter': {'bleu': [], 'rouge1': [], 'rougeL': []}}

    for item in tqdm(data):

        task = item['task']
        bleu, rouge, corpus_bleu_score = compute_metrics(item['responses'], item['gt_ref_ans'])
        task_metrics[task]['bleu'].append(bleu)
        task_metrics[task]['rouge1'].append(rouge['rouge1'])
        task_metrics[task]['rougeL'].append(rouge['rougeL'])


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
        print(f"  BLEU: {metrics['bleu']}")
        print(f"  ROUGE-1: {metrics['rouge1']}")
        print(f"  ROUGE-L: {metrics['rougeL']}")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Compute BLEU and ROUGE metrics for task responses.")
    
    # Define input and output arguments
    parser.add_argument('--input_path', type=str, help="Path to the input JSON file.", 
    default="")
    parser.add_argument('--output_path', type=str, help="Path to save the output JSON file with metrics.",
    default="ngram_metrics.json")

    # Parse the arguments
    args = parser.parse_args()

    # Execute the main function with provided arguments
    main(args.input_path, args.output_path)