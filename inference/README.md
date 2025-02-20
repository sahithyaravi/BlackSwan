# Inference Guide

### Before you begin:
In both the files, there is a base path that points to the folder where the videos (with and without blacked out frames for task 1 and task 2) are stored. You have to change it as required. Also, make a copy of `mcq_list.json` and `VAR_data.json` and store them in the folders where you launch the python code. 

## LlavaNeXT

1. Clone `https://github.com/LLaVA-VL/LLaVA-NeXT`
2. Copy the `video_demo-llavanext.py` file into the `playground/demo` folder (rename and replace `video_demo.py`)
3. Follow `https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/docs/LLaVA-NeXT-Video.md` to run:

You can use: 
```
bash scripts/video/demo/video_demo.sh lmms-lab/LLaVA-NeXT-Video-7B-DPO vicuna_v1 32 2 True ./data/llava_video/video-chatgpt/evaluation/Test_Videos/v_Lf_7RurLgp0.mp4
```

## VideoLlama2

1. Clone and setup `https://github.com/DAMO-NLP-SG/VideoLLaMA2`
2. Copy `inference-videollama2.py` into the folder
3. Run the python script
