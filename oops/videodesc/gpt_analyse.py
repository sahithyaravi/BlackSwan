import json

with open('mcq_list.json', 'r') as f:
    data = json.load(f)

with open('mcq_list_both.json', 'r') as f:
    data_both = json.load(f)

for d in data:

    if d['mcq_task'] == 1:

        db = None
        for d_both in data_both:
            if d_both['set_id'] == d['set_id'] and d_both['id'] == d['id'] and d_both['mcq_id'] == d['mcq_id']:
                db = d_both

        is_correct_d = False    
        if d['predicted'].startswith('A') and d['mcq_label'] == 0:
            is_correct_d = True
        elif d['predicted'].startswith('B') and d['mcq_label'] == 1:
            is_correct_d = True
        elif d['predicted'].startswith('C') and d['mcq_label'] == 2:
            is_correct_d = True
        
        is_correct_db = False
        if db['predicted'].startswith('A') and db['mcq_label'] == 0:
            is_correct_db = True
        elif db['predicted'].startswith('B') and db['mcq_label'] == 1:
            is_correct_db = True
        elif db['predicted'].startswith('C') and db['mcq_label'] == 2:
            is_correct_db = True

        if (not is_correct_d) and (not is_correct_db):
            print(f"Both correct: {d['set_id']}, {d['id']}, {d['mcq_id']}")


