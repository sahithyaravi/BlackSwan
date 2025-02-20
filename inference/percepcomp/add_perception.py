import json
import csv
import pandas as pd

with open('../../data/mcq_list_all_gpt_subset_t1.json', 'r') as f:
    mcq_list = json.load(f)

annotated_df_path = 'perception_data.csv'

df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]
ids_completed = []
for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    set_id = task_data[0]['CellData']
    video_id = task_data[1]['CellData']

    for mcq in mcq_list:
        if mcq['set_id'] == set_id and mcq['preevent'] == video_id:
            mcq['pc_preevent'] = submitted_data['#txtArea1a2']

        if mcq['set_id'] == set_id and mcq['postevent'] == video_id:
            mcq['pc_postevent'] = submitted_data['#txtArea1a2']


    
annotated_df_path = 'comprehension_data.csv'

df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]
ids_completed = []
for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    set_id = task_data[0]['CellData'].split('/')[-1]
    preevent = task_data[1]['CellData']
    postevent = task_data[2]['CellData']

    #print(set_id, preevent, postevent, submitted_data['#txtArea1a2'])

    for mcq in mcq_list:
        if mcq['set_id'] == set_id and mcq['preevent'] == preevent and mcq['postevent'] == postevent:
            mcq['pc_comp'] = submitted_data['#txtArea1a2']


with open('../../data/mcq_list_all_gpt_subset_t1_pc.json', 'w') as f:
    json.dump(mcq_list, f, indent=4)

