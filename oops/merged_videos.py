import json
from moviepy.editor import VideoFileClip, concatenate_videoclips, ColorClip
import os
import argparse

parser = argparse.ArgumentParser(description="Generate Images")
parser.add_argument("--set", type=str, help="Set", required=True)
args = parser.parse_args()

set_id = args.set

# Load the JSON file
with open(f"{set_id}.json", "r") as file:
    data = json.load(file)

#if path doesnt exist make folder
if not os.path.exists(f"{set_id}_merged"):
    os.makedirs(f"{set_id}_merged")

for annot in data:

    # Load the video clips
    video1 = VideoFileClip(f"{set_id}/{annot['preevent']}")
    video2 = VideoFileClip(f"{set_id}/{annot['event']}")
    video3 = VideoFileClip(f"{set_id}/{annot['postevent']}")

    duration = video2.duration
    size = video2.size

    # Create a black color clip with the same duration and size
    black_clip = ColorClip(size, color=(0, 0, 0), duration=duration)

    # Set the frame rate to match the original video
    black_clip = black_clip.set_fps(video2.fps)

    # Concatenate the video clips
    final_video = concatenate_videoclips([video1, black_clip, video3])

    # Write the result to a file
    final_video.write_videofile(f"{set_id}_merged/{str(annot['index'])}_D_merged.mp4")

    # Concatenate the video clips
    final_video2 = concatenate_videoclips([video1, video2, video3])

    # Write the result to a file
    final_video2.write_videofile(f"{set_id}_merged/{str(annot['index'])}_E_merged.mp4")