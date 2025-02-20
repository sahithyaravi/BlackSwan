import transformers
import torch
from tqdm import tqdm
import json

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("/model-weights/Meta-Llama-3.1-8B", local_files_only=True, device_map="auto")
# model = AutoModelForCausalLM.from_pretrained("/model-weights/Meta-Llama-3.1-8B", local_files_only=True, device_map="auto")
model_id = "/model-weights/Meta-Llama-3.1-70B-Instruct"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": {"load_in_8bit": True},
        "low_cpu_mem_usage": True,
    },
    device_map="auto",
)

with open("../../data/mcq_list_v9p2_v2_gpt_human_desc.json", "r") as f:
    mcq_list = json.load(f)


correct = 0
failed = 0
total = 0

for q in tqdm(mcq_list[:30]):

    if q['mcq_task'] == 1:

        predesc = q['preevent_desc']
        postdesc = q['postevent_desc']

        A = q["mcq_options"][0]
        B = q["mcq_options"][1]
        C = q["mcq_options"][2]

        question = f"Beginning: {predesc} \n End: {postdesc} \n Question: Which of the following descriptions indicate what is most likely happening in the video?\nA. {A}\nB. {B}\nC. {C}.\n Only answer with the letter associated with the correct answer, A, B, or C."

        system_prompt = """You will answer Multiple Choice questions. Each question will include two video descriptions, one for the beginning and one for the end of the video. 
        You will be asked to reason about what is happening in between the beginning and the end of the video, based on the descriptions. You can only reply A, B, or C.
        
        Here is an example:
        Beginning: A girl in a white shirt and black pants swings on a rope swing. She is swinging off a cliff and she will land in a river or lake. It is a sunny day.
        End: A young woman is swimming towards the shore in what looks to be a lake.  She is using a modified breast stroke. There is a cut of tree stump in front of her. She is speaking in a language that does not sound like English.
        Question: Which of the following descriptions indicate what is most likely happening in the video?
        A. The woman cheers as she successfully reaches the other side by hanging onto the swinging rope.
        B. The woman lets go of the swinging rope and intentionally goes into the water.
        C. The woman will stand on the rock, reaching up to catch a frisbee.
        Correct Answer: B
        """


        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipe(
            messages,
            max_new_tokens=25,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        assistant_response = outputs[0]["generated_text"][-1]["content"]
        
        out = assistant_response.strip()
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

        with open(f'out/mcq_list_1shot_{model_id.split("/")[-1]}.json', 'w') as f:
            json.dump(mcq_list, f, indent=4)

print("MCQ Task 1", model_id)
print(f"Correct: {correct}, Failed: {failed}, Total: {total}")
print(f"Accuracy: {correct/total}")