import json

with open('/h/adityac/projects/VAR/data/VAR_Data.json', 'r') as f:
    data = json.load(f)

easy = 0
medium = 0
hard = 0
hard_list=[]

for annot in data:
    count_yes = annot['task3_checkboxes'].count('yes')
    #count_no = annot['task2_checkboxes'].count('no')

    if count_yes >= 2:
        easy += 1
        annot['difficulty'] = 'easy'
    elif count_yes == 1:
        medium += 1
        annot['difficulty'] = 'medium'
    else:
        hard += 1
        annot['difficulty'] = 'hard'
        # annot['preevent'] = annot['videos_url']+annot['preevent']
        # annot['event'] = annot['videos_url']+annot['event']
        # annot['postevent'] = annot['videos_url']+annot['postevent']
        hard_list.append(annot)
    

print(f"More than one Yes: {easy}")
print(f"Only one Yes: {medium}")
print(f"No Yes (aka hard): {hard}")

# with open('hard.json', 'w') as f:
#     json.dump(hard_list, f, indent=4)

# Save to a new CSV file
with open('/h/adityac/projects/VAR/data/VAR_Data.json', 'w') as f:
    json.dump(data, f, indent=4)