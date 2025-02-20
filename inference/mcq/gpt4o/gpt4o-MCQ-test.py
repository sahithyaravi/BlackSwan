import json
from tqdm import tqdm

def eval_ans(label, options, out):

    answer = "failed"

    if out.startswith('Answer:'):
        out = out.split('Answer:')[1].strip()
    
    out = out.replace('\n', '').replace('\r', '')

    if out == 'A' or out.startswith('(A)') or out.startswith('A.') or (options[0].lower() in out.lower()) or ('A.' in out):
        if label == 0:
            answer = "correct"
        else:
            answer = "incorrect"

    elif out == 'B' or out.startswith('(B)') or out.startswith('B.') or (options[1].lower() in out.lower()) or ('B.' in out):
        if label == 1:
            answer = "correct"
        else:
            answer = "incorrect"
        
    elif out == 'C' or out.startswith('(C)') or out.startswith('C.') or (options[2].lower() in out.lower()) or ('C.' in out):
        if label == 2:
            answer = "correct"
        else:
            answer = "incorrect"

    else: 
        print(out)

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

with open(f"mcq-output.jsonl", "r") as f:
    output = f.readlines()
    output = [json.loads(o) for o in output]

with open("../../../data/mcq_list_all_gpt.json") as f:
    data = json.load(f)

for q in tqdm(data[:10]):

    curr_task = q["mcq_task"]

    for o in output:
        if f"request-{q['set_id']}-{q['id']}-{q['mcq_id']}" == o["custom_id"]:
            q['predicted'] = o['response']['body']['choices'][0]['message']['content']
            break

    ans = eval_ans(q["mcq_label"], q["mcq_options"], q['predicted'])

    if curr_task == 1:
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