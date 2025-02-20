from openai import OpenAI
import json
import random
client = OpenAI(api_key=API_KEY)
from tqdm import tqdm

def do_gpt_convert(main, reference, use_alt=True):
    prompt = f"Your task is to reframe the Given sentence in the same form as the Reference sentence. Do NOT change the meaning of the given sentence. Do not add any content from the reference sentence into the answer sentence. The answer sentence must be different from the reference sentence. Only return the answer sentence.\n \
                Example: \n Reframe the following sentence based on the reference sentence: \n \
                Given: A boy wearing glasses is standing in front of a door. \n \
                Reference: The kid will try to throw the basketball in the basket and someone will open the door \n \
                Answer: The kid will stand in front of the door wearing glasses. \n \
                In this example, 'The boy' will change to 'The kid', and the tense should change to future tense.\n \n \
                Task: \n Reframe the following sentence based on the reference sentence: \n \
                Given:  '{main}' \n \
                Reference: '{reference}' \n \
                Answer:"

    prompt2 = f"Your task is to reframe the Given sentence in the same form as the Reference sentence. Do NOT change the meaning of the given sentence. Do not add any content from the reference sentence into the answer sentence. The answer sentence must be different from the reference sentence. Only return the answer sentence.\n \
                Example: \n Reframe the following sentence based on the reference sentence: \n \
                Given: A boy wearing glasses is standing in front of a door. \n \
                Reference: The kid will try to throw the basketball in the basket and someone will open the door \n \
                Answer: The kid will stand in front of the door wearing glasses. \n \
                In this example, 'The boy' will change to 'The kid', and the tense should change to future tense. \
                Moreover, the length of the sentence remains similar to the given sentence.\n \n \
                Task: \n Reframe the following sentence based on the reference sentence: \n \
                Given:  '{main}' \n \
                Reference: '{reference}' \n \
                Answer:"

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt if not use_alt else prompt2}
        ]
    )

    #print(completion.choices[0].message.content)
    return completion.choices[0].message.content

with open('../../data/mcq_list_all.json') as f:
    data = json.load(f)

# with open('mcq_list_v2_gpt.json', 'r') as f:
#     mcq_list = json.load(f)

mcq_list = []

total_count = 0

for annot in tqdm(data):

    non_empty = [a for a in annot['mcq_options'] if a != '']
    if len(non_empty) < 3:
        continue

    if annot['preevent_caption'] in annot['mcq_options']:

        total_count += 1
        
        # Find the caption in the options
        index = annot['mcq_options'].index(annot['preevent_caption'])

        # Find the wrong answer index
        wrong_answer_index = [i for i in range(3) if (i != index) and (i != annot['mcq_label'])][0]

        completed = False
        try_count = 0
        fail_count = 0
        while not completed:
            if try_count < 2:
                new_answer = do_gpt_convert(annot['preevent_caption'], annot['mcq_options'][wrong_answer_index])
            else:
                new_answer = do_gpt_convert(annot['preevent_caption'], annot['mcq_options'][wrong_answer_index], use_alt=False)
            new_answer = new_answer.replace('\n', ' ').replace('\r', ' ').replace("'", '').replace('"', '')    
            
            if new_answer not in annot['mcq_options'] and len(new_answer.split(' ')) > (len(annot['preevent_caption'].split(' '))-3) and len(new_answer.split(' ')) < (len(annot['preevent_caption'].split(' '))+3):
                completed = True
                annot['mcq_options'][index] = new_answer
                annot['gpt_modified'] = 'yes'
            elif try_count > 15:
                new_answer = annot['preevent_caption']
                annot['mcq_options'][index] = new_answer
                annot['gpt_modified'] = 'no_failed'
                completed = True
                fail_count += 1
            else:
                print('Trying again', new_answer)
                try_count += 1

        # annot['mcq_options'][index] = new_answer
        # annot['gpt_modified'] = 'yes'

        mcq_list.append(annot)

    else:
        annot['gpt_modified'] = 'no'
        mcq_list.append(annot)


    with open('../../data/mcq_list_all_gpt.json', 'w') as f:
        json.dump(mcq_list, f, indent=4)

print(total_count)
print(fail_count)


