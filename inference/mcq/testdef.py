import json
from tqdm import tqdm

def eval_ans(label, out):

    answer = "failed"

    if out.startswith('Answer:'):
        out = out.split('Answer:')[1].strip()
    
    out = out.replace('\n', '').replace('\r', '').lower()

    if out == 'yes' or out.startswith('yes') or out.startswith('yes.'): #or (options[0].lower() in out.lower()) or ('A.' in out):
        if label == 'yes':
            answer = "correct"
        else:
            answer = "incorrect"

    elif out == 'no' or out.startswith('no') or out.startswith('no.'): #or (options[1].lower() in out.lower()):
        if label == 'no':
            answer = "correct"
        else:
            answer = "incorrect"

    # else: 
    #     print(out)

    return answer

correct_t1 = 0
wrong_t1 = 0
failed_t1 = 0
total_t1 = 0
correct_t2 = 0
wrong_t2 = 0
failed_t2 = 0
total_t2 = 0

model = 'llavavideo7B'

with open(f"../../results/def_list_all_subset_gpt4o.json", "r") as f:
    mcq_list = json.load(f)

with open(f"../../data/VAR_Data.json", "r") as f:
    data = json.load(f)

for q in tqdm(mcq_list):
    
    if not "predicted" in q:
        continue

    annot = None
    for d in data:
        if d["set_id"] == q["set_id"] and d['id'] == q['id']:
            annot = d
            break        
            
    curr_task = q["def_task"]

    ans = eval_ans(q["def_label"], q['predicted'])

    if curr_task == 1:# and annot['difficulty'] != 'hard':
        if ans == "correct":
            correct_t1 += 1
        elif ans == "incorrect":
            wrong_t1 += 1
        else:
            failed_t1 += 1
        total_t1 += 1
    else:
        if ans == "correct":
            correct_t2 += 1
        elif ans == "incorrect":
            wrong_t2 += 1
        else:
            failed_t2 += 1
        total_t2 += 1

print('Task 1')
print(f"Correct: {correct_t1} \nWrong: {wrong_t1} \nFailed: {failed_t1} \nTotal: {total_t1}")
print(f"Accuracy: {round(correct_t1/total_t1, 3)}")

print('Task 2')
print(f"Correct: {correct_t2} \nWrong: {wrong_t2} \nFailed: {failed_t2} \nTotal: {total_t2}")
print(f"Accuracy: {round(correct_t2/total_t2, 3)}")