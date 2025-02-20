# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import os
import json
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="float16", attn_implementation="flash_attention_2", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()
#model = model.to(torch.bfloat16)

with open('/fs01/home/adityac/projects/VAR/data/VAR_Data.json', 'r') as f:
    VAR_Data = json.load(f)

with open("/fs01/home/adityac/projects/VAR/results/def_list_all_llavavideo7B.json", "r") as f:
    mcq_list = json.load(f)

base_path = '/fs01/home/adityac/projects/VAR/oops/'

for q in tqdm(mcq_list):

    curr_task = q["def_task"]
    hypo = q["exp"]

    if "predicted" in q:
        continue

    if curr_task == 1:
        video_path = os.path.join(base_path, f"{q['set_id']}_merged", f"{q['index']}_D_merged.mp4")
        question = f"Hypothesis: {hypo}\n Given the video clip, does this hypothesis hold? Answer yes or no."
    else:
        video_path = os.path.join(base_path, f"{q['set_id']}_merged", f"{q['index']}_E_merged.mp4")   
        question = f"Hypothesis: {hypo}\n Given the video clip, does this hypothesis hold? Answer yes or no."

    max_frames_num = 32
    video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
    video = [video]
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following question related to this video."
    #question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n"+question
    question = DEFAULT_IMAGE_TOKEN+f"\n{question}"
    #print(question)

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    # print(input_ids.dtype)
    # print(video[0].dtype)

    cont = model.generate(
        input_ids,
        images=video,
        modalities= ["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=10,
    )

    #print(cont)

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    out = text_outputs.strip()

    q["predicted"] = out

    with open('/fs01/home/adityac/projects/VAR/results/def_list_all_llavavideo7B.json', 'w') as f:
        json.dump(mcq_list, f, indent=4)
