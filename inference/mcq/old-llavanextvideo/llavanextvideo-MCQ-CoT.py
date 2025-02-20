import av
import numpy as np
import json
import torch
from tqdm import tqdm
import os
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)

processor = LlavaNextVideoProcessor.from_pretrained(model_id)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


with open("../../data/mcq_list_v9-11_v2_gpt.json", "r") as f:
    mcq_list = json.load(f)

correct = 0
failed = 0
total = 0
base_path = '/fs01/home/adityac/projects/VAR/oops/'

for q in tqdm(mcq_list):

    if q['mcq_task'] == 1:
        A = q["mcq_options"][0]
        B = q["mcq_options"][1]
        C = q["mcq_options"][2]

        
        video_path = os.path.join(base_path, f"{q['set_id']}", f"{q['preevent']}")
        #question = f"Which of the following descriptions indicate what happened in the hidden (black) frames of the video?\nA. {A}\nB. {B}\nC. {C}\nAnswer with the option's letter from the given choices directly."
        question = f"This is the beginning of the scene. Describe this video in one sentence."

        # define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video"},
                    ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        container = av.open(video_path)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip = read_video_pyav(container, indices)
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        out = processor.decode(output[0], skip_special_tokens=True)
        out = out.split("ASSISTANT:")[-1].strip()
        #print(out)
       
        conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": out},
                ],
            }
        )

        video_path = os.path.join(base_path, f"{q['set_id']}", f"{q['postevent']}")
        question = f"This is the end of the scene. Describe this video in one sentence."

        conversation.append(
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "video"},
                    ],
            }
        )

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        container = av.open(video_path)

        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        clip2 = read_video_pyav(container, indices)
        inputs_video = processor(text=prompt, videos=[clip,clip2], padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        out = processor.decode(output[0], skip_special_tokens=True)
        out = out.split("ASSISTANT:")[-1].strip()
        #print(out)
       
        conversation.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": out},
                ],
            }
        )

        # video_path = os.path.join(base_path, f"{q['set_id']}_merged", f"{q['index']}_D_merged.mp4")
        # question = f"Now, compare the end to the beginning. Describe what changed, and what could have caused this change."

        # conversation.append(
        #     {

        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": question},
        #             {"type": "video"},
        #             ],
        #     }
        # )

        # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # container = av.open(video_path)

        # # sample uniformly 8 frames from the video, can sample more for longer videos
        # total_frames = container.streams.video[0].frames
        # indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        # clip3 = read_video_pyav(container, indices)
        # inputs_video = processor(text=prompt, videos=[clip,clip2,clip3], padding=True, return_tensors="pt").to(model.device)

        # output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        # out = processor.decode(output[0], skip_special_tokens=True)
        # out = out.split("ASSISTANT:")[-1].strip()
        # #print(out)
       
        # conversation.append(
        #     {
        #         "role": "assistant",
        #         "content": [
        #             {"type": "text", "text": out},
        #         ],
        #     }
        # )

        question = f"Having seen the beginning and the end, which of the following descriptions indicate what is most likely happening in the video?\n(A) {A}\n(B) {B}\n(C) {C}\nAnswer with the option's letter from the given choices directly."

        conversation.append(
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    #{"type": "video"},
                    ],
            }
        )

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # container = av.open(video_path)

        # # sample uniformly 8 frames from the video, can sample more for longer videos
        # total_frames = container.streams.video[0].frames
        # indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        # clip4 = read_video_pyav(container, indices)
        inputs_video = processor(text=prompt, videos=[clip,clip2], padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        out = processor.decode(output[0], skip_special_tokens=True)
        out = out.split("ASSISTANT:")[-1].strip()

        pred = None
        if out.startswith('A') or out.startswith('(A)'):
            pred = 0
        elif out.startswith('B') or out.startswith('(B)'):
            pred = 1
        elif out.startswith('C') or out.startswith('(C)'):
            pred = 2
        else:
            failed += 1
        
        if pred == q["mcq_label"]: 
            correct += 1
            
        total += 1

        q["predicted"] = out
        q["conversation"] = conversation

        with open(f'out/mcq_list_CoT-wo_{model_id.split("/")[-1]}.json', 'w') as f:
            json.dump(mcq_list, f, indent=4)

print(f"Correct: {correct}, Failed: {failed}, Total: {total}")
print(f"Accuracy: {correct/total}")