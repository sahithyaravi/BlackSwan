import csv

with open('human-mcq-t1-all.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

print(data[0])

total = 0
correct = 0
completed = []
time_diffs = []
corrects = []

workers = []

completed_hits = {}

for d in data:

    if d['WorkerId'] not in workers:
        workers.append(d['WorkerId'])

    if d['Answer.mcq_answer'] == 'option_1':
        given_ans = 0
    elif d['Answer.mcq_answer'] == 'option_2':
        given_ans = 1
    elif d['Answer.mcq_answer'] == 'option_3':
        given_ans = 2
    
    true_ans = int(d['Input.mcq_label'])

    if d['HITId'] in completed_hits.keys():
        completed_hits[d['HITId']].append(1 if true_ans == given_ans else 0)
    else:
        completed_hits[d['HITId']] = [1] if true_ans == given_ans else [0]

    if true_ans == given_ans:
        correct += 1

    else:
        if d['Input.set_id'] == 'oops_val_v9' and int(d['Input.id']) > 50:
            print(d['Input.set_id'], d['Input.id'], d['Input.mcq_id'], true_ans, given_ans)
    total += 1

    completed.append(d['HITId'])
    time_diffs.append(int(d['WorkTimeInSeconds']))
    corrects.append(1 if true_ans == given_ans else 0)


print(len(completed_hits.keys()))
#Take the max of the list of correctness for each HIT
corrects2 = [max(completed_hits[k]) for k in completed_hits.keys()]
print('total:', len(corrects2))
print('Good:', len([c for c in corrects2 if c == 1]))

print(f"Workers:{workers}")

print(f"Correct: {correct}",
        f"Total: {total}")


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
plt.show()