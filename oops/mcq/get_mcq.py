import pandas as pd
import csv
import json
import random

annotated_df_path="../oops/annotations/VAR_Data.json"

with open(annotated_df_path) as f:
    data = json.load(f)

mcq_list = []

for annot in data:

    if annot['multivideo'] == 'yes':
        continue

    base_annot = {
        'set_id': annot['set_id'],
        'id': annot['id'],
        'index': annot['index'],
        'videos_url': annot['videos_url'],
        'frames_url': annot['frames_url'],
        'preevent': annot['preevent'],
        'event': annot['event'],
        'postevent': annot['postevent']
    }

    mcq_annot_id = 0

    wrong_answers = [annot['task1_gt'][i] for i in range(3) if annot['task2_checkboxes'][i] == 'no']

    for i in range(3):
        if annot['task2_checkboxes'][i] == 'no':
            for wrong_answer in wrong_answers:
                if random.random() > 0.5:
                    mcq_options = [wrong_answer, annot['task2_gt'][i]]
                    mcq_label = 1
                else:
                    mcq_options = [annot['task2_gt'][i], wrong_answer]
                    mcq_label = 0
                new_annot = base_annot.copy()
                new_annot['mcq_task'] = 1
                new_annot['mcq_id'] = mcq_annot_id
                new_annot['mcq_options'] = mcq_options
                new_annot['mcq_label'] = mcq_label
                mcq_list.append(new_annot)
                mcq_annot_id += 1
    
    wrong_answers = [annot['task2_gt'][i] for i in range(3) if annot['task3_checkboxes'][i] == 'no']

    if annot['task3_gt'] != '':
        for wrong_answer in wrong_answers:
            if random.random() > 0.5:
                mcq_options = [wrong_answer, annot['task3_gt']]
                mcq_label = 1
            else:
                mcq_options = [annot['task3_gt'], wrong_answer]
                mcq_label = 0
            new_annot = base_annot.copy()
            new_annot['mcq_task'] = 2
            new_annot['mcq_annot_id'] = mcq_annot_id
            new_annot['mcq_options'] = mcq_options
            new_annot['mcq_label'] = mcq_label
            mcq_list.append(new_annot)
            mcq_annot_id += 1

with open('mcq_list.json', 'w') as f:
    json.dump(mcq_list, f, indent=4)

print(len(mcq_list))