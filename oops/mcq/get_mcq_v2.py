import pandas as pd
import csv
import json
import random

annotated_df_path="../../data/VAR_Data_caption.json"

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
        'postevent': annot['postevent'],
        'preevent_caption': annot['preevent_caption'],
    }

    mcq_annot_id = 0

    wrong_answers = [annot['task1_gt'][i] for i in range(3) if annot['task2_checkboxes'][i] == 'no']
    wrong_answers_sets = []
    if len(wrong_answers) == 0:
        continue
    elif len(wrong_answers) == 1:
        wrong_answers_sets = [(wrong_answers[0], annot['preevent_caption'])]
    elif len(wrong_answers) == 2:
        wrong_answers_sets = [(wrong_answers[0], wrong_answers[1])]
    elif len(wrong_answers) == 3:
        wrong_answers_sets = [(wrong_answers[0], wrong_answers[1]), (wrong_answers[0], wrong_answers[2]), (wrong_answers[1], wrong_answers[2])]

    for i in range(3):
        if annot['task2_checkboxes'][i] == 'no':
            for wrong_answer_set in wrong_answers_sets:
                
                mcq_options = [wrong_answer_set[0], wrong_answer_set[1], annot['task2_gt'][i]]
                mcq_label = [0,0,1]
                # Shuffle options and labels in the same order
                zipped = list(zip(mcq_options, mcq_label))
                random.shuffle(zipped)
                mcq_options, mcq_label = zip(*zipped)
                mcq_label = mcq_label.index(1)
                new_annot = base_annot.copy()
                new_annot['mcq_task'] = 1
                new_annot['mcq_id'] = mcq_annot_id
                new_annot['mcq_options'] = mcq_options
                new_annot['mcq_label'] = mcq_label
                mcq_list.append(new_annot)
                mcq_annot_id += 1
    
    annot['task2_gt'] = [annot['task2_gt'][i] if annot['task2_gt'][i] != "" else annot['task1_gt'][i] for i in range(3)]
    wrong_answers = [annot['task2_gt'][i] for i in range(3) if annot['task3_checkboxes'][i] == 'no']
    wrong_answers_sets = []
    if len(wrong_answers) == 0:
        continue
    elif len(wrong_answers) == 1:
        wrong_answers_sets = [(wrong_answers[0], annot['preevent_caption'])]
    elif len(wrong_answers) == 2:
        wrong_answers_sets = [(wrong_answers[0], wrong_answers[1])]
    elif len(wrong_answers) == 3:
        wrong_answers_sets = [(wrong_answers[0], wrong_answers[1]), (wrong_answers[0], wrong_answers[2]), (wrong_answers[1], wrong_answers[2])]

    if annot['task3_gt'] != '':
        for wrong_answer_set in wrong_answers_sets:
            
            mcq_options = [wrong_answer_set[0], wrong_answer_set[1], annot['task3_gt']]
            mcq_label = [0,0,1]
            # Shuffle options and labels in the same order
            zipped = list(zip(mcq_options, mcq_label))
            random.shuffle(zipped)
            mcq_options, mcq_label = zip(*zipped)
            mcq_label = mcq_label.index(1)

            new_annot = base_annot.copy()
            new_annot['mcq_task'] = 2
            new_annot['mcq_id'] = mcq_annot_id
            new_annot['mcq_options'] = mcq_options
            new_annot['mcq_label'] = mcq_label
            mcq_list.append(new_annot)
            mcq_annot_id += 1

with open('../../data/mcq_list_all.json', 'w') as f:
    json.dump(mcq_list, f, indent=4)

print(len(mcq_list))