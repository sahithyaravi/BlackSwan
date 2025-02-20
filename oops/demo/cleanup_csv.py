import pandas as pd
from collections import Counter
import json

annotated_df_path="v4.csv"
df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]
data = []
for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']
    
    set_id = task_data[0]['CellData']
    annot_id = task_data[1]['CellData']
    
    # print("############ TASK1 RESULT ############")
    video_id = task_data[3]['CellData']
    video_link1 = f"https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops/{set_id}/{video_id}"

    # print("############ TASK2 RESULT ############")
    video_id = task_data[4]['CellData']
    video_link2 = f"https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops/{set_id}/{video_id}"

    # print("############ TASK3 RESULT ############")
    video_id = task_data[5]['CellData']
    video_link3 = f"https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops/{set_id}/{video_id}"

    annot = {
        'set_id': set_id,
        'id': annot_id,
        'video1link': video_link1,
        'video2link': video_link2,
        'video3link': video_link3,
        'explanation1a': submitted_data['#txtArea1a1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'explanation1b': submitted_data['#txtArea1b1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'explanation1c': submitted_data['#txtArea1c1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'explanation2a': submitted_data['#txtArea2a1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'checkbox2a': submitted_data['explanation2a'],
        'explanation2b': submitted_data['#txtArea2b1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'checkbox2b': submitted_data['explanation2b'],
        'explanation2c': submitted_data['#txtArea2c1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'checkbox2c': submitted_data['explanation2c'],
        'explanation3a': submitted_data['#txtArea3a1'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
        'checkbox3a': submitted_data['explanation3a'],
        'checkbox3b': submitted_data['explanation3b'],
        'checkbox3c': submitted_data['explanation3c'],
        'multivideo': submitted_data['radio_multi'],
        'optional_feedback': submitted_data['#fb-container3'].strip().replace('\n', ' ').replace('\r', ' ').replace(',', '-'),
    }
    if submitted_data['radio_multi'] == 'yes':
        print(annot_id, task_data[2]['CellData'])
    if submitted_data['#fb-container3'] != '':
        print(annot_id, task_data[2]['CellData'], submitted_data['#fb-container3'])

    if submitted_data['explanation2a'] == 'yes' and submitted_data['explanation2b'] == 'yes' and submitted_data['explanation2c'] == 'yes':
        print(annot_id, task_data[2]['CellData'], 'Simple: all yes')

    data.append(annot)

# Save to a new CSV file
df = pd.DataFrame(data)
#convert id to numeric sort by annot_id
df['id'] = pd.to_numeric(df['id'])
df = df.sort_values(by=['set_id', 'id'])
df.to_csv("videos.csv", index=False, header=False)

print(Counter([d['multivideo'] for d in data]))