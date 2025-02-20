import json


with open('mcq_list_v2_gpt.json') as f:
    data = json.load(f)

print("Length of data: ", len(data))

print("GPT-modded", len([d for d in data if d['gpt_modified'] == 'yes']))