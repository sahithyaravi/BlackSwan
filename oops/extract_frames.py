# Take a video clip and extract all the frame from it as images
import cv2
import os
import csv
import glob
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Generate Images")
parser.add_argument("--set", type=str, help="Set", required=True)
args = parser.parse_args()

set_name = args.set
frames_save_path = f'{set_name}_frames/'
NUM_FRAMES = 10


def extract_frames_fps(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Crete output path if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get the frame rate of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Calculate how many frames to skip to get 4 frames per second
    frame_skip = fps // 2
    
    # Initialize frame counter
    frame_count = 0
    
    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # Check if frame was read successfully
        if not ret:
            break
        
        # If it's time to save a frame
        if frame_count % frame_skip == 0:
            # Save the frame
            frame_filename = f"{output_path}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        # Increment frame counter
        frame_count += 1
    
    # Release the video object
    video.release()


def extract_frames(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Crete output path if not exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate indices to uniformly select 4 frames
    frame_indices = np.linspace(0, total_frames - 1, num=NUM_FRAMES, dtype=int)
    
    # Initialize index counter
    frame_count = 0

    # Initialize frame number
    frame_number = 1
    
    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = video.read()
        
        # Check if frame was read successfully
        if not ret:
            break
        
        # If the current frame index matches one of the selected indices
        if frame_count in frame_indices:
            # Save the frame
            frame_filename = f"{output_path}/frame_{frame_number}.jpg"
            cv2.imwrite(frame_filename, frame)
            
            # Increment frame number
            frame_number += 1

            # Remove the saved index from the list
            frame_indices = np.delete(frame_indices, np.where(frame_indices == frame_count))
        
        # If we've already saved all 4 frames, break out of the loop
        if len(frame_indices) == 0:
            break
        
        # Increment frame counter
        frame_count += 1
    
    # Release the video object
    video.release()


def autoextract(json_file=None):
    if json_file is None:
        json_file = f'{set_name}.json'

    # Open csv file
    with open(json_file, 'r') as file:
        dataset = json.load(file)

    for row in dataset:

        pre_path = os.path.join(set_name,row['preevent'])
        pre_frames_path = os.path.join(frames_save_path,row['preevent'][0:-4])
        extract_frames(pre_path, pre_frames_path)

        event_path = os.path.join(set_name,row['event'])
        event_frames_path = os.path.join(frames_save_path,row['event'][0:-4])
        extract_frames(event_path, event_frames_path)

        post_path = os.path.join(set_name,row['postevent'])
        post_frames_path = os.path.join(frames_save_path,row['postevent'][0:-4])
        extract_frames(post_path, post_frames_path)

if __name__ == "__main__":
    autoextract()