import json
import csv

with open('../../../data/VAR_Data_eval_filtered_subset.json', 'r') as f:
    data = json.load(f)


for d in data:

    d['merged_path'] = f"{d['videos_url'][:-1]}_merged/{d['index']}_E_merged.mp4"
    del d['task1_gt']
    del d['task2_gt']
    del d['task3_gt']
    del d['task2_checkboxes']
    del d['task3_checkboxes']
    del d['multivideo']
    del d['optional_feedback']
    del d['assignment_id']
    del d['participant_id']
    del d['start_time']
    del d['end_time']
    del d['difficulty']

# Save as csv
with open(f'VAR_Data_filtered_subet_human_gen_alltasks.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    for annot in data:
        writer.writerow(annot)