import json

with open('Var_Data_gemini_gen.json', 'r') as f:
    data = json.load(f)

for item in data:
    item['task2_responses'] = [i['task2_response'] for i in item['task2_responses'] if i != "" and 'task2_response' in i]
    item['task3_responses'] = [i['task3_response'] for i in item['task3_responses'] if i != "" and 'task3_response' in i]

with open('VAR_Data_gemini_gen.json', 'w') as f:
    json.dump(data, f, indent=4)