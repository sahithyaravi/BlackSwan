import json

def compute_task_avg_scores(file_path):
    # Load JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize dictionaries to sum scores and count entries per task
    task_sums = {}
    task_counts = {}

    # Iterate over each entry in the JSON
    for entry in data:
        for key, value in entry.items():
            # Check for 'task' and 'avg_score' in the key to identify task average scores
            if 'task' in key and 'avg_score' in key:
                # Extract the task name (e.g., "task1", "task2") from the key
                task_name = key.split('_avg_score')[0]

                # Accumulate sums and counts for each task
                if task_name not in task_sums:
                    task_sums[task_name] = 0
                    task_counts[task_name] = 0
                task_sums[task_name] += value
                task_counts[task_name] += 1

    # Calculate the overall average for each task's average score
    task_avg_scores = {task: task_sums[task] / task_counts[task] for task in task_sums}

    return task_avg_scores

# Example usage
file_path = '/h/sahiravi/VAR/results/VAR_Data_llavavideo7B_match_llm_allscores.json'  # Replace with your actual JSON file path
average_scores = compute_task_avg_scores(file_path)
print(average_scores)