import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, save_path):
    """Download a file from a URL to a specified save path."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download: {url}")
    except:
         print(f"Failed with error download: {url}")


def download_media_files(data, local_base_path):
    for item in tqdm(data[1200:]):
        videos_url = item["videos_url"]
        frames_url = item["frames_url"]
        index = item["index"]

        # Define the URLs to download
        media_urls = {
            "preevent": f"{videos_url}{item['preevent']}",
            "event": f"{videos_url}{item['event']}",
            "postevent": f"{videos_url}{item['postevent']}",
            "merged_D": f"{videos_url[:-1]}_merged/{index}_D_merged.mp4",
            "merged_E": f"{videos_url[:-1]}_merged/{index}_E_merged.mp4",
        }

        frame_urls = {
            "preevent_frames": [f"{frames_url}{index}_A_preevent/frame_{i}.jpg" for i in range(1, 11)],
            "event_frames": [f"{frames_url}{index}_B_event/frame_{i}.jpg" for i in range(1, 11)],
            "postevent_frames": [f"{frames_url}{index}_C_postevent/frame_{i}.jpg" for i in range(1, 11)],
        }

        # Download each media file
        for key, url in media_urls.items():
            domain = "https://ubc-cv-sherlock.s3.us-west-2.amazonaws.com"
            relative_path = url.replace(domain, "")
            save_path = f"{local_base_path}{relative_path}"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            download_file(url, save_path)

        # Download each frame image
        for key, urls in frame_urls.items():
            for url in urls:
                # print("local base path", local_base_path)
                relative_path = url.replace(domain, "")
                save_path = f"{local_base_path}{relative_path}"
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                download_file(url, save_path)
# Load JSON file
with open("/h/sahiravi/VAR/data/VAR_Data.json", "r") as file:
    data = json.load(file)

# Define the local base path to save files
local_base_path = "/h/sahiravi/scratch/var_data/"

print(len(data))
# # Download the media files
download_media_files(data, local_base_path)