import json
import csv
import random

with open('VAR_Data_v9p2_caption.json', 'r') as f:
    data = json.load(f)

out = []

for d in data:
    out.append({'videos_url': d['videos_url'], 'preevent': d['preevent'], 'postevent': d['postevent']})

random.shuffle(out)

with open('videodesc_both_v9p2.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['videos_url', 'preevent', 'postevent'])
    writer.writeheader()
    writer.writerows(out)