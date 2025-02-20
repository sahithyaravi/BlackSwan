import json
from evaluate import load
from scipy.optimize import linear_sum_assignment

def bertscore(s1, s2, score='precision'):
    bert = load("bertscore")
    predictions = [s1]
    references = [s2]
    results = bert.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    return results[score][0]

def bleu(s1, s2):
    bleu = load("bleu")
    results = bleu.compute(predictions=[s1], references=[s2])
    return results['bleu']

def rouge(s1, s2, score='rouge1'):
    rouge = load("rouge")
    results = rouge.compute(predictions=[s1], references=[s2])
    return results[score]

# Use bipartitle matching to find the best match between two lists of sentences using the score function. Use scipy.optimize.linear_sum_assignment to solve the assignment problem.
def match(set1, set2, score_func):
    scores = [[score_func(s1, s2) for s2 in set2] for s1 in set1]
    row_ind, col_ind = linear_sum_assignment(scores, maximize=True)
    matches = [(set1[i], set2[j], scores[i][j]) for i, j in zip(row_ind, col_ind)]
    return matches
    
def eval_helper(set1, set2, mode, score_func):

    if mode == "outcome":
        set1 = [s['Hypothesis'] for s in set1 if s['Hypothesis'] != '']
        set2 = [s['Outcome'] for s in set2 if s['Outcome'] != '']
    elif mode == "explanation":
        set1 = [s['Explanation'] for s in set1 if s['Explanation'] != '']
        set2 = [s['Explanation'] for s in set2 if s['Explanation'] != '']
    elif mode == "both":
        set1 = [s['Hypothesis']+' '+s['Explanation'] for s in set1 if s['Hypothesis'] != '' and s['Explanation'] != '']
        set2 = [s['Outcome']+' '+s['Explanation'] for s in set2 if s['Outcome'] != '' and s['Explanation'] != '']

    #print(set1, set2)
    matches = match(set1, set2, score_func)

    avg_score = sum([m[2] for m in matches]) / len(matches)
    max_score = max([m[2] for m in matches])

    return avg_score, max_score, matches


def run_eval(dataset, mode, score_func):

    avg_scores_task1 = []
    avg_scores_task2 = []
    avg_scores_task3 = []

    max_scores_task1 = []
    max_scores_task2 = []
    max_scores_task3 = []

    low_score = 0.8

    print("Dataset size:", len(dataset))

    for data in dataset:

        # Task 1:
        set1 = data['task1_gt']
        set2 = data['task1_responses']

        avg_score, max_score, _ = eval_helper(set1, set2, mode, score_func)
        avg_scores_task1.append(avg_score)
        max_scores_task1.append(max_score)

        # Task 2:
        set1 = data['task2_gt']
        set2 = data['task2_responses']

        avg_score, max_score, matches = eval_helper(set1, set2, mode, score_func)
        avg_scores_task2.append(avg_score)
        max_scores_task2.append(max_score)

        for m in matches:
            if m[2] < low_score:
                print(m)
                low_score = m[2]

        # Task 3:
        set1 = data['task3_gt']
        set2 = data['task3_responses']

        avg_score, max_score, _ = eval_helper(set1, set2, mode, score_func)
        avg_scores_task3.append(avg_score)
        max_scores_task3.append(max_score)

    print("Avg max score for task 1:", sum(max_scores_task1)/len(max_scores_task1))
    print("Avg max score for task 2:", sum(max_scores_task2)/len(max_scores_task2))
    print("Avg max score for task 3:", sum(max_scores_task3)/len(max_scores_task3))

    print("Avg avg score for task 1:", sum(avg_scores_task1)/len(avg_scores_task1))
    print("Avg avg score for task 2:", sum(avg_scores_task2)/len(avg_scores_task2))
    print("Avg avg score for task 3:", sum(avg_scores_task3)/len(avg_scores_task3))



with open("oops_val_v1_mf.json", 'r') as f:
    dataset = json.load(f)


run_eval(dataset, 'both', bertscore)
