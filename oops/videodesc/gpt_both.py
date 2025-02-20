from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI(api_key=API_KEY)


with open("mcq_list_v9p2_v2_gpt_desc.json", "r") as f:
    mcq_list = json.load(f)


correct = 0
failed = 0
total = 0

for q in tqdm(mcq_list):

    if q['mcq_task'] == 1:

        predesc = q['preevent_desc']
        postdesc = q['postevent_desc']
        bothdesc = q['both_desc']

        A = q["mcq_options"][0]
        B = q["mcq_options"][1]
        C = q["mcq_options"][2]

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You will answer a Multiple Choice question. A question will include two video descriptions, one for the beginning and one for the end of the video. Moreover, you will receive a description for the `difference`, which says what changed between the beginning and end of the video clip.  You will be asked to choose the correct answer based on the descriptions. You can only reply A, B, or C."},
                {
                    "role": "user",
                    "content": f"Beginning: {predesc} \n End: {postdesc} \n Difference: {bothdesc} \n Question: Which of the following descriptions indicate what is most likely happening in the video?\nA. {A}\nB. {B}\nC. {C}.\n Only answer with the letter associated with the correct answer, A, B, or C."
                }
            ]
        )

        #print(completion.choices[0].message)
        out = completion.choices[0].message.content.strip()
        print(out)

        if out.startswith('A') or out.startswith('B') or out.startswith('C'):
            if out.startswith('A') and q["mcq_label"] == 0:
                correct += 1
            elif out.startswith('B') and q["mcq_label"] == 1:
                correct += 1
            elif out.startswith('C') and q["mcq_label"] == 2:
                correct += 1
        else:
            failed += 1
        
        total += 1

        q["predicted"] = out

        with open('mcq_list_both.json', 'w') as f:
            json.dump(mcq_list, f, indent=4)


print(f"Correct: {correct}, Failed: {failed}, Total: {total}")
print(f"Accuracy: {correct/total}")