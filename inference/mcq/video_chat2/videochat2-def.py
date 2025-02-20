from utils.config import Config
import io

from models import VideoChat2_it_hd_mistral
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm
import os

import decord
import time
decord.bridge.set_bridge("torch")

from videochat2utils import load_video, get_sinusoid_encoding_table, ask, answer, eval_ans

config_file = "configs/config_mistral_hd.json"
cfg = Config.from_file(config_file)

# load stage2 model
cfg.model.vision_encoder.num_frames = 4
model = VideoChat2_it_hd_mistral(config=cfg.model)

# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)
model.mistral_model = get_peft_model(model.mistral_model, peft_config)

state_dict = torch.load("/fs01/home/adityac/scratch/videochat2_hd_mistral_7b_stage4.pth", "cuda")

if 'model' in state_dict.keys():
    msg = model.load_state_dict(state_dict['model'], strict=False)
else:
    msg = model.load_state_dict(state_dict, strict=False)
#print(msg)

model = model.to(torch.device(cfg.device))
model = model.eval()


with open('/fs01/home/adityac/projects/VAR/data/VAR_Data.json', 'r') as f:
        VAR_Data = json.load(f)

with open("/h/adityac/projects/VAR/results/def_list_all_videochat2.json", "r") as f:
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

    # num_frame = 8
    num_frame = 16
    # resolution = 384
    resolution = 224
    # hd_num = 6
    hd_num = 12
    padding = False
    vid, msg = load_video(
        video_path, num_segments=num_frame, return_msg=True, resolution=resolution,
        hd_num=hd_num, padding=padding
    )
    new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
    model.vision_encoder.encoder.pos_embed = new_pos_emb

    #print(msg)
        
    # The model expects inputs of shape: T x C x H x W
    T_, C, H, W = vid.shape
    video = vid.reshape(1, T_, C, H, W).to("cuda:0")

    img_list = []
    with torch.no_grad():
        image_emb, _, _ = model.encode_img(video, "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, answer the question.\n")
    #     image_emb, _, _ = model.encode_img(video, "")

    img_list.append(image_emb[0])

    chat = EasyDict({
        "system": "",
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
    # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg} [/INST]"])
    ask(question, chat)

    llm_message = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=5, print_res=False)[0]
    out = llm_message.strip()
    #print(out)
    
    q["predicted"] = out

    with open('/fs01/home/adityac/projects/VAR/results/def_list_all_videochat2.json', 'w') as f:
        json.dump(mcq_list, f, indent=4)

# vid_path = "/h/adityac/projects/Llava/Ask-Anything/video_chat2/example/yoga.mp4"
# # vid_path = "./demo/example/jesse_dance.mp4"


# # num_frame = 8
# num_frame = 16
# # resolution = 384
# resolution = 224
# # hd_num = 6
# hd_num = 12
# padding = False
# vid, msg = load_video(
#     vid_path, num_segments=num_frame, return_msg=True, resolution=resolution,
#     hd_num=hd_num, padding=padding
# )
# new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
# model.vision_encoder.encoder.pos_embed = new_pos_emb

# print(msg)
    
# # The model expects inputs of shape: T x C x H x W
# T_, C, H, W = vid.shape
# video = vid.reshape(1, T_, C, H, W).to("cuda:0")

# img_list = []
# with torch.no_grad():
#     image_emb, _, _ = model.encode_img(video, "Watch the video and answer the question.")
# #     image_emb, _, _ = model.encode_img(video, "")

# img_list.append(image_emb[0])

# chat = EasyDict({
#     "system": "",
#     "roles": ("[INST]", "[/INST]"),
#     "messages": [],
#     "sep": ""
# })

# chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
# # chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg} [/INST]"])
# ask("Describe the video in details.", chat)

# llm_message = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, print_res=True)[0]
# print(llm_message)