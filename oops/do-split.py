import os
import json
import csv
import glob
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from scenedetect import detect, ContentDetector, split_video_ffmpeg, AdaptiveDetector
import numpy as np

# Read all filenames in the directory
set_data = "/datasets/oops/oops_video/val"
files = os.listdir(set_data)

# Load the JSON file
with open("/datasets/oops/annotations/transition_times.json", "r") as file:
    data = json.load(file)

# To manage our data, we split it into multiple sets for the purposes of data collection. 
# These sets get merged later.
setnumsd = list(range(1, 17))

for setnum in setnumsd:

    set_name = f'oops_val_v{str(setnum)}'
    save_base = set_name

    trim_ends = 0.17

    #load all existing json files
    file_list = glob.glob("*.json")
    data_prev_fnames = []
    for file in file_list:
        with open(file, "r") as file:
            data_prev = json.load(file)
            data_prev_fnames += [d['original'] for d in data_prev]

    print("Skipping: ", len(data_prev_fnames))

    # Create the videos folder if it doesn't exist
    os.makedirs(save_base, exist_ok=True)

    dataset = []

    print("Total files:", len(files))

    skip_trans = 0
    skip_badsplit = 0
    skip_bad2split = 0
    skip_badtime = 0
    skip_final = 0

    done = 0
    ncount = 0
    for index, file in enumerate(tqdm(files)):
        # Check if the file is in the json file
        f_name = file.split(".")[0]
        if (f_name in data) and (f_name not in data_prev_fnames):
            # Get the transition times
            transition_times = data[f_name]
            split = [t for t in transition_times['t'] if t > 0.1]
            #print(split)
            if len(split) == 0:
                skip_trans += 1
                continue
            if np.std(split) > 3:
                if len(split) >= 2:
                    new_split = []
                    # Compute pairwise std, if one pair has std < 0.5, use that
                    for i in range(len(split)):
                        for j in range(i+1, len(split)):
                            if np.std([split[i], split[j]]) <= 4:
                                new_split = [split[i], split[j]]
                                break
                    if len(new_split) == 0:
                        skip_trans += 1
                        continue
                    else:
                        split = new_split
                else:
                    skip_trans += 1
                    continue

            if np.std(split) <= 0.5:
                split_t = sum(split) / len(split)
            else:
                split = [t for t in split if t > 1]
                if len(split) == 0:
                    skip_badtime += 1
                    continue
                split_t = min(split)
            
            modified_file = "no"
            
            scene_list = detect(os.path.join(set_data, file), AdaptiveDetector(window_width=5))
            if len(scene_list) > 2:
                if scene_list[0][1].get_seconds() - scene_list[0][0].get_seconds() < 1:
                    scene_list.pop(0)
                    modified_file = "cropped"
                if scene_list[-1][1].get_seconds() - scene_list[-1][0].get_seconds() < 1:
                    scene_list.pop(-1)
                    modified_file = "cropped"
                
                if len(scene_list) > 2:
                    skip_badsplit += 1
                    continue

            video_clip = VideoFileClip(os.path.join(set_data, file))
            if len(scene_list) == 2:
                # First check if split_t is in first or second part
                if split_t > scene_list[0][1].get_seconds() and scene_list[0][1].get_seconds() <= 1.5:
                    # We can crop the first part of the video
                    video_clip = video_clip.subclip(scene_list[0][1].get_seconds(), video_clip.duration)
                    modified_file = "del_beginning"
                    split_t = split_t - scene_list[0][1].get_seconds()
                elif split_t < scene_list[0][1].get_seconds() and (scene_list[1][1].get_seconds()-scene_list[1][0].get_seconds()) <= 1.5:
                    # We can crop the second part of the video
                    video_clip = video_clip.subclip(0, scene_list[0][1].get_seconds())
                    modified_file = "del_end"
                elif split_t-0.5 < scene_list[0][1].get_seconds() and split_t+0.5 > scene_list[0][1].get_seconds():
                    modified_file = "allowed"
                else:
                    skip_bad2split += 1
                    continue

            if video_clip.duration >= 3 and (split_t*0.8-trim_ends) >= 1 and (video_clip.duration*0.8-split_t*0.8) >= 1 and ((video_clip.duration-trim_ends)-video_clip.duration*0.8) >= 1:

                # Preevent
                preevent_clip = video_clip.subclip(trim_ends, split_t*0.8)
                preevent_name = f"{index + 1}_A_preevent.mp4"
                preevent_path = os.path.join(save_base, preevent_name)
                preevent_clip.write_videofile(preevent_path, temp_audiofile=False)

                # Event
                event_clip = video_clip.subclip(split_t*0.8, video_clip.duration*0.8)
                event_name = f"{index + 1}_B_event.mp4"
                event_path = os.path.join(save_base, event_name)
                event_clip.write_videofile(event_path, temp_audiofile=False)

                # PostEvent
                postevent_clip = video_clip.subclip(video_clip.duration*0.8, video_clip.duration-trim_ends)
                postevent_name = f"{index + 1}_C_postevent.mp4"
                postevent_path = os.path.join(save_base, postevent_name)
            
                postevent_clip.write_videofile(postevent_path, temp_audiofile=False)

                video_clip.close()

                dataset.append({
                    "set": set_name,
                    "id": done+1,
                    "index": index+1,
                    "preevent": preevent_name,
                    "event": event_name,
                    "postevent": postevent_name,
                    "transition": split_t,
                    "is_modified": modified_file,
                    "original": f_name
                })

                print(f"Processed video {index + 1}")
                done += 1
            else:
                skip_final += 1

        if done == 100:
            break

    print("Total videos processed: ", done)
    print("Skipped due to no transition: ", skip_trans)
    print("Skipped due to bad split: ", skip_badsplit)
    print("Skipped due to bad time: ", skip_badtime)
    print("Skipped due to bad2 split: ", skip_bad2split)
    print("Skipped due to final: ", skip_final)
     
    with open(f"{set_name}.json", "w") as file:
        json.dump(dataset, file, indent=4)
        
    with open(f"{set_name}.csv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=dataset[0].keys())
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)