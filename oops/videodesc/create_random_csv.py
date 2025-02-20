import json
import csv
import random

with open('VAR_Data_v9p2_caption.json', 'r') as f:
    data = json.load(f)

out = []

for d in data:
    out.append({'set_id': d['set_id'], 'video_id': d['preevent']})
    out.append({'set_id': d['set_id'], 'video_id': d['postevent']})

random.shuffle(out)

with open('videodesc_v9p2.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['set_id', 'video_id'])
    writer.writeheader()
    writer.writerows(out)