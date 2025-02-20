import json

with open("/fs01/home/adityac/projects/VAR/data/mcq_list_all_gpt_videochat2.json", "r") as f:
    mcq_list = json.load(f)

task1 = 0
task2 = 0

for q in mcq_list:
    if q["mcq_task"] == 1:
        task1 += 1
    else:
        task2 += 1

print(task1, task2)