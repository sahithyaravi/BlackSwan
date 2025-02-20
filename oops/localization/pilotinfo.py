import pandas as pd
from collections import Counter
import numpy as np

import json

data = []

annotated_df_path = 'data1.csv'
annotated_df_path2 = 'data2.csv'

df = pd.read_csv(annotated_df_path)
df = df[df['Submitted Data'].notna()]

df2 = pd.read_csv(annotated_df_path2)
df2 = df2[df2['Submitted Data'].notna()]

agree = 0
total = 0

for idx, row in df.iterrows():
    task_data  = json.loads(row['Task Data'])['RowData']
    submitted_data  = json.loads(row['Submitted Data'])['Data']['taskData']
    index = task_data[1]['CellData']

    # Find index in df2
    for idx2, row2 in df2.iterrows():
        task_data2  = json.loads(row2['Task Data'])['RowData']
        submitted_data2  = json.loads(row2['Submitted Data'])['Data']['taskData']
        index2 = task_data2[1]['CellData']
        if index2 == index:
            break
            
    # Check for agreement in submitted data
    t1 = submitted_data['t1']
    t2 = submitted_data2['t1']

    agrees = False
    if np.std([float(t1), float(t2)]) < 0.5 or abs(float(t1) - float(t2)) <= 1:
        agree += 1
        agrees = True
    total += 1

    print(index, submitted_data, submitted_data2, agrees)

print(agree, total, agree/total)
        
            