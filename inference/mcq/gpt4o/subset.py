import json
import random

with open("../../../data/def_list_all.json", 'r') as f:
    data = json.load(f)

new_items_task1 = []
new_items_task2 = []

count_task1 = 0
count_task2 = 0

random.seed(10)
random.shuffle(data)

for item in data:

    if item['def_task'] == 1 and count_task1 < 150:
        new_items_task1.append(item)
        count_task1 += 1

    if item['def_task'] == 2 and count_task2 < 150:
        new_items_task2.append(item)
        count_task2 += 1

print(len(new_items_task1))
print(len(new_items_task2))

with open("../../../data/def_list_all_subset.json", 'w') as f:
    json.dump(new_items_task1 + new_items_task2, f, indent=4)