import pandas as pd
from collections import Counter
import json

with open("mcq_list_v9p2_v2_gpt_desc.json", "r") as f:
    mcq_list = json.load(f)

data = []
for q in mcq_list:

    if q['mcq_task'] == 2:
        continue
    
    set_id = q['set_id']
    annot_id = q['id']
    
    # print("############ TASK1 RESULT ############")
    video_link1 = q['videos_url']+q['preevent']

    # print("############ TASK2 RESULT ############")
    video_link2 = q['videos_url']+q['event']

    # print("############ TASK3 RESULT ############")
    video_link3 = q['videos_url']+q['postevent']

    annot = {
        'set_id': set_id,
        'id': annot_id,
        'mcq_task': q['mcq_task'],
        'mcq_id': q['mcq_id'],
        'video1link': video_link1,
        'video2link': video_link2,
        'video3link': video_link3,
        'preevent_desc': q['preevent_desc'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'postevent_desc': q['postevent_desc'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'both_desc': q['both_desc'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'A': q["mcq_options"][0].replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'B': q["mcq_options"][1].replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'C': q["mcq_options"][2].replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'correctans': "A" if q["mcq_label"] == 0 else "B" if q["mcq_label"] == 1 else "C",
    }
    
    data.append(annot)

# Save to a new CSV file
df = pd.DataFrame(data)
#convert id to numeric sort by annot_id
df['id'] = pd.to_numeric(df['id'])
df = df.sort_values(by=['set_id', 'id'])
df.to_csv("mcq_demo.csv", index=False, header=False)
