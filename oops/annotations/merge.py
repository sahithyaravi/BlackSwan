import pandas as pd
from collections import Counter

import json

data = []

def extract_data(annotated_df_path):
    df = pd.read_csv(annotated_df_path)
    df = df[df['Submitted Data'].notna()]
    ids_completed = []
    for idx, row in df.iterrows():
        task_data  = json.loads(row['Task Data'])['RowData']
        submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']

        if len(task_data) == 7:
            set_id = 'oops_val_v1'
            is_modified = 'no'
        elif len(task_data) == 8:
            set_id = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'set'][0]
            is_modified = "no"
        else:
            set_id = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'set'][0]
            is_modified = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'is_modified'][0]

        annot_id = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'id'][0]
        index = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'index'][0]
        preevent = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'preevent'][0]
        event = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'event'][0]
        postevent = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'postevent'][0]
        transition = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'transition'][0]
        original = [task_data[i]['CellData'] for i in range(len(task_data)) if task_data[i]['ColumnHeader'] == 'original'][0]

        annot = {
            'set_id': set_id,
            'id': annot_id,
            'index': index,
            'videos_url': f"https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops/{set_id}/",
            'frames_url': f"https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops/{set_id}_frames/",
            'preevent': preevent,
            'event': event,
            'postevent': postevent,
            'transition': transition,
            'is_modified': is_modified,
            'original': original,
            'task1_gt': [submitted_data['#txtArea1a1'].strip().replace('\n', ' ').replace('\r', ' '), 
                        submitted_data['#txtArea1b1'].strip().replace('\n', ' ').replace('\r', ' '), 
                        submitted_data['#txtArea1c1'].strip().replace('\n', ' ').replace('\r', ' ')],
            'task2_checkboxes': [submitted_data['explanation2a'], submitted_data['explanation2b'], submitted_data['explanation2c']],
            'task2_gt': [submitted_data['#txtArea2a1'].strip().replace('\n', ' ').replace('\r', ' '),
                        submitted_data['#txtArea2b1'].strip().replace('\n', ' ').replace('\r', ' '),
                        submitted_data['#txtArea2c1'].strip().replace('\n', ' ').replace('\r', ' ')],
            'task3_checkboxes': [submitted_data['explanation3a'], submitted_data['explanation3b'], submitted_data['explanation3c']],
            'task3_gt': submitted_data['#txtArea3a1'].strip().replace('\n', ' ').replace('\r', ' '),
            'multivideo': '' if 'radio_multi' not in submitted_data else submitted_data['radio_multi'],
            'optional_feedback': submitted_data['#fb-container3'].strip().replace('\n', ' ').replace('\r', ' '),
            'assignment_id': row['AssignmentId'],
            'participant_id': row['ParticipantId'],
            'start_time': row['StartTime (America/Vancouver)'] if 'StartTime (America/Vancouver)' in row else row['StartTime (America/Chicago)'],
            'end_time': row['CompletionTime (America/Vancouver)'] if 'CompletionTime (America/Vancouver)' in row else row['CompletionTime (America/Chicago)'],
        }

        if annot_id not in ids_completed:
            data.append(annot)
            ids_completed.append(annot_id)


extract_data("v1p1.csv")
extract_data("v1p2.csv")
extract_data("v2p1.csv")
extract_data("v2p2.csv")
extract_data("v3p1.csv")
extract_data("v3p2.csv")
extract_data("v4p1.csv")
extract_data("v4p2.csv")
extract_data("v5p1.csv")
extract_data("v5p2.csv")
extract_data("v6p1.csv")
extract_data("v6p2.csv")
extract_data("v7p1.csv")
extract_data("v7p2.csv")
extract_data("v8p1.csv")
extract_data("v8p2.csv")
extract_data("v9p1.csv")
extract_data("v9p2.csv")
extract_data("v10p1.csv")
extract_data("v10p2.csv")
extract_data("v11p1.csv")
extract_data("v11p2.csv")
extract_data("v12p1.csv")
extract_data("v12p2.csv")
extract_data("v13p1.csv")
extract_data("v13p2.csv")
extract_data("v14p1.csv")
extract_data("v14p2.csv")
extract_data("v15p1.csv")
extract_data("v15p2.csv")
extract_data("v16p1.csv")
extract_data("v16p2.csv")
extract_data("v17p1.csv")
extract_data("v18p1.csv")

print('Total: ', len(data))


# Save to a new CSV file
with open('VAR_Data.json', 'w') as f:
    json.dump(data, f, indent=4)
