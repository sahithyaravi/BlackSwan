import pandas as pd
from IPython.display import Video, HTML
from collections import Counter
from datetime import datetime
import random

import json

def find_match(task_data, sdf):
    for idx, row in sdf.iterrows():
        if json.loads(row['Task Data'])['RowData'] == task_data:
            return idx
    return None

def time_diff(t1, t2):

    # Define the format for the date time strings
    date_format = "%m/%d/%Y %I:%M:%S %p"

    # Convert the strings to datetime objects
    t1 = datetime.strptime(t1, date_format)
    t2 = datetime.strptime(t2, date_format)

    # Calculate the difference in seconds
    time_difference = (t2 - t1).total_seconds()
    return time_difference

annotated_df_path="human-mcq-t2-all.csv"
df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]

annotated_df_path="human-mcq-t2-all-2.csv"
df2 = pd.read_csv(annotated_df_path)
df2 = df2[df2['Submitted Data'].notna()]

correct = 0
correct_1 = 0
correct_2 = 0
total = 0

workers = {}
time_diffs = []
corrects = []

annotators = {}

with open('../../../data/VAR_Data_caption.json') as f:
    vardata = json.load(f)

for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    match_idx = find_match(task_data, df2)
    if match_idx is None:
        print("not found")
        continue
    match_row = df2.loc[match_idx]
    match_submitted_data  = json.loads(match_row['Submitted Data'])['Data']['taskData']

    #print(task_data)
    true_ans = int(task_data[11]['CellData'])
    given_ans = 3
    if submitted_data['mcq_answer'] == 'option_1':
        given_ans = 0
    elif submitted_data['mcq_answer'] == 'option_2':
        given_ans = 1
    elif submitted_data['mcq_answer'] == 'option_3':
        given_ans = 2

    given_ans_2 = 3
    if match_submitted_data['mcq_answer'] == 'option_1':
        given_ans_2 = 0
    elif match_submitted_data['mcq_answer'] == 'option_2':
        given_ans_2 = 1
    elif match_submitted_data['mcq_answer'] == 'option_3':
        given_ans_2 = 2

    #print(true_ans, given_ans, given_ans_2)

    worker1 = row['ParticipantId']
    worker2 = match_row['ParticipantId']

    if worker1 not in workers:
        workers[worker1] = []

    if worker2 not in workers:
        workers[worker2] = []

    
    if (true_ans == given_ans):
        correct_1 += 1
        workers[worker1].append(1)
    else:
        workers[worker1].append(0)

    if (true_ans == given_ans_2):
        correct_2 += 1
        workers[worker2].append(1)
    else:
        workers[worker2].append(0)

    if (true_ans == given_ans) or (true_ans == given_ans_2):
        correct += 1
    total += 1

    time_diffs.append(time_diff(row['StartTime (America/Chicago)'], row['CompletionTime (America/Chicago)']))
    corrects.append(1 if (true_ans == given_ans) else 0)
    time_diffs.append(time_diff(match_row['StartTime (America/Chicago)'], match_row['CompletionTime (America/Chicago)']))
    corrects.append(1 if (true_ans == given_ans_2) else 0)


    if not((true_ans == given_ans)) and not((true_ans == given_ans_2)):
        d = {t['ColumnHeader']: t['CellData'] for t in task_data}
        print(d['set_id'], d['id'], d['mcq_id'], row['ParticipantId'])
        options = [d['option1'], d['option2'], d['option3']]
        options[true_ans] = "y - "+options[true_ans]
        options[given_ans] = "n - "+options[given_ans]
        for o in options:
            print(o)
        for vd in vardata:
            if vd['set_id'] == d['set_id'] and vd['id'] == d['id']:
                annotators[vd['participant_id']] = annotators.get(vd['participant_id'], 0) + 1


        
print(f"Correct 1: {correct_1}",
        f"Correct 2: {correct_2}",
        f"Correct: {correct}",
      f"Total: {total}")
print(f"Accuracy 1: {correct_1/total}")
print(f"Accuracy 2: {correct_2/total}")
print(f"Accuracy: {correct/total}")

print(annotators)



for worker in workers:
    if len(workers[worker]) > 5:
        print(f"{worker}: {round(sum(workers[worker])/len(workers[worker]), 2)}, {len(workers[worker])}")


import matplotlib.pyplot as plt

# Colors: green for correct, red for wrong
colors = ['green' if c == 1 else 'red' for c in corrects]

# Create a scatter plot
plt.scatter(time_diffs, range(len(time_diffs)), c=colors, s=100, edgecolors='k')

# Adding labels and title
plt.xlabel('Time Difference (seconds)')
plt.xlim(0, 100)
plt.ylabel('Sample Index (ignore this)')
plt.title('Correlation between Time Difference and Correctness')

# Display the plot
#plt.show()
plt.close()