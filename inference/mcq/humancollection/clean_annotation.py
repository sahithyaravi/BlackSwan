import json
import csv
import pandas as pd

with open('../../../results/VAR_Data_eval_filtered_subset_human.json') as f:
    vardata = json.load(f)


annotated_df_path="human_annot_gen_t3.csv"
df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]

for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

    print(task_data)
    print(submitted_data)

    # Find the corresponding annot
    for d in vardata:
        if d['set_id'] == task_data[0]['CellData'] and d['id'] == task_data[1]['CellData']:
            d['task3_responses'] = submitted_data['#txtArea1a1']#, submitted_data['#txtArea1b1'], submitted_data['#txtArea1c1']]
            #d['answer_pid'] = row['Participant ID']


#Save var data
with open(f'../../../results/VAR_Data_eval_filtered_subset_human.json', 'w') as f:
    json.dump(vardata, f, indent=4)
