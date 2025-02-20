import pandas as pd
from collections import Counter

import json

# Load the annotated data
with open('mcq_list_v9p2_v2_gpt.json', 'r') as f:
    data = json.load(f)

annotated_df_path = 'v9p2_25words.csv'

df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]

for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    video_id = task_data[1]['CellData']
    for d in data:
        if d['preevent'] == video_id:
            d['preevent_desc'] = submitted_data['#txtArea1a2']
        if d['postevent'] == video_id:
            d['postevent_desc'] = submitted_data['#txtArea1a2']


# Save to a new CSV file
with open('mcq_list_v9p2_v2_gpt_desc.json', 'w') as f:
    json.dump(data, f, indent=4)
