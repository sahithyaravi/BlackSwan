import json
import csv

with open('../../data/mcq_list_all_gpt_subset_t1.json', 'r') as f:
    mcq_list = json.load(f)

video_list = []

for annot in mcq_list:

    video_list.append({
        'set_id': annot['set_id'],
        'video_id': annot['preevent']
    })

    # video_list.append({
    #     'set_id': annot['set_id'],
    #     'video_id': annot['event']
    # })

    video_list.append({
        'set_id': annot['set_id'],
        'video_id': annot['postevent']
    })

with open('perception-mcq_list_all_gpt_subset_t1.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=video_list[0].keys())
    writer.writeheader()
    for annot in video_list:
        writer.writerow(annot)