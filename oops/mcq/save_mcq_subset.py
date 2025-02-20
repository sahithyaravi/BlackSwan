import json
import csv
import pandas as pd

annotated_df_path="mcq_list_all_gpt_human_t2.csv"
df = pd.read_csv(annotated_df_path)

newset = []

with open('../../data/mcq_list_all_gpt.json') as f:
    vardata = json.load(f)

for idx, row in df.iterrows():

    for d in vardata:
        if str(d["set_id"]) == str(row['set_id']) and str(d['id']) == str(row['id']) and str(d['mcq_id']) == str(row['mcq_id']):
            newset.append(d)

print(len(newset))
with open('mcq_list_all_gpt_subset_t2.json', 'w') as f:
    json.dump(newset, f, indent=4)