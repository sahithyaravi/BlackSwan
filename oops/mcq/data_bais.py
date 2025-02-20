import json
import numpy as np

with open('../../data/mcq_list_all_gpt.json') as f:
    data = json.load(f)


correct_ans = []
wrong_ans = []

for q in data:
    if q['mcq_task'] == 1:
        correct = q['mcq_options'][q['mcq_label']]
        correct_ans.append(correct)
        q['mcq_options'].remove(correct)
        wrong_ans = wrong_ans + q['mcq_options']

    
print(np.mean([len(a.split(' ')) for a in correct_ans]))
print(np.mean([len(a.split(' ')) for a in wrong_ans]))


print("Length of data: ", len(data))

print("GPT-modded", len([d for d in data if d['gpt_modified'] == 'yes']))