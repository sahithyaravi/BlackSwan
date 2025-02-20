import pandas as pd
from collections import Counter

import json

# Load the annotated data
with open('mcq_list_v9p2_v2_gpt_desc.json', 'r') as f:
    data = json.load(f)

annotated_df_path = 'v9p2_both.csv'

df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]

for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    path = task_data[0]['CellData']
    preevent = task_data[1]['CellData']
    postevent = task_data[2]['CellData']
    for d in data:
        if d['preevent'] == preevent and d['postevent'] == postevent:
            d['both_desc'] = submitted_data['#txtArea1a2']


# Save to a new CSV file
with open('mcq_list_v9p2_v2_gpt_desc.json', 'w') as f:
    json.dump(data, f, indent=4)
