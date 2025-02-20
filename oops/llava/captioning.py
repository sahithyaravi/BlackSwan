from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image
import requests
import json
from tqdm import tqdm

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# specify how to quantize the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, device_map="auto")

url_base = 'https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com/oops_val_v1_frames/'
subsets = ['preevent', 'event', 'postevent']


with open('oops_val_v1_captions.json', 'r') as f:
    dataset = json.load(f)

for data in tqdm(dataset):
    for sub in subsets:
        if sub + '_caption' in data:
            print('skipping', data['id'], sub)
            continue

        image_urls = [url_base + f'{data[sub][0:-4]}/frame_{i}.jpg' for i in range(1, 5)]
        #print(image_urls)
        image_list = []
        for url in image_urls:
            image = Image.open(requests.get(url, stream=True).raw)
            image_list.append(image)

        prompt = "[INST] <image>\nDescribe this frame in one sentence: [/INST]"

        inputs = processor([prompt]*len(image_list), image_list, return_tensors="pt").to("cuda")

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=20)

        caption_list = []
        for i in range(len(image_list)):
            caption = processor.decode(output[i], skip_special_tokens=True).split("[/INST]")[1].strip()
            caption_list.append(caption)

        data[sub + '_caption'] = caption_list
    
        # save back to json
        with open('oops_val_v1_captions.json', 'w') as f:
            json.dump(dataset, f, indent=4)