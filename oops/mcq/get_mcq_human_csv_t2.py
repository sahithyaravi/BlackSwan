import csv
import json
import random

with open('../../data/mcq_list_all_gpt.json') as f:
    data = json.load(f)

mcq_list = []

random.shuffle(data)

for annot in data:

    if annot['mcq_task'] == 2:
        new_annot = {
            'set_id': annot['set_id'],
            'id': annot['id'],
            'videos_url': annot['videos_url'],
            'preevent': annot['preevent'],
            'event': annot['event'],
            'postevent': annot['postevent'],
            'mcq_task': annot['mcq_task'],
            'mcq_id': annot['mcq_id'],
            'option1': annot['mcq_options'][0].replace(',', '').replace('\n', ' ').replace('\r', ' '),
            'option2': annot['mcq_options'][1].replace(',', '').replace('\n', ' ').replace('\r', ' '),
            'option3': annot['mcq_options'][2].replace(',', '').replace('\n', ' ').replace('\r', ' '),
            'mcq_label': annot['mcq_label']
        }
        if new_annot['option1'] == "" or new_annot['option2'] == "":
            continue

        mcq_list.append(new_annot)

    if len(mcq_list) == 150:
        break

with open('mcq_list_all_gpt_human_t2.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=mcq_list[0].keys())
    writer.writeheader()
    for annot in mcq_list:
        writer.writerow(annot)
