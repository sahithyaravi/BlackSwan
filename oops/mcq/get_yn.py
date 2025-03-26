import pandas as pd
import csv
import json
import random

annotated_df_path="../../data/VAR_Data_caption.json"

with open(annotated_df_path) as f:
    data = json.load(f)

mcq_list = []
yes_count1 = 0
no_count1 = 0
yes_count2 = 0
no_count2 = 0

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

    def_annot_id = 0

    for i, exp in enumerate(annot['task1_gt']):
        if exp == "":
            continue

        new_annot = base_annot.copy()
        new_annot['def_task'] = 1
        new_annot['def_id'] = def_annot_id
        new_annot['exp'] = exp
        new_annot['def_label'] = annot['task2_checkboxes'][i]

        if new_annot['def_label'] == 'yes':
            yes_count1 += 1
        else:
            no_count1 += 1

        mcq_list.append(new_annot)
        def_annot_id += 1

    for i, exp in enumerate(annot['task2_gt']):
        if exp == "":
            continue

        new_annot = base_annot.copy()
        new_annot['def_task'] = 2
        new_annot['def_id'] = def_annot_id
        new_annot['exp'] = exp
        new_annot['def_label'] = annot['task3_checkboxes'][i]

        if new_annot['def_label'] == 'yes':
            yes_count2 += 1
        else:
            no_count2 += 1

        mcq_list.append(new_annot)
        def_annot_id += 1

        
with open('../../data/def_list_all.json', 'w') as f:
    json.dump(mcq_list, f, indent=4)

print(len(mcq_list), yes_count1, no_count1, yes_count2, no_count2)