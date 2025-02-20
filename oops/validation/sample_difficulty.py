import json
import random
import os
import pandas as pd

# Load JSON data from a file
with open('VAR_Data-v6-11.json', 'r') as file:
    data = json.load(file)

# Filter examples by difficulty
easy_examples = [entry for entry in data if entry.get('difficulty') == 'easy']
medium_examples = [entry for entry in data if entry.get('difficulty') == 'medium']
hard_examples = [entry for entry in data if entry.get('difficulty') == 'hard']

# Randomly sample 10 examples from each category
sampled_easy = random.sample(easy_examples, min(10, len(easy_examples)))
sampled_medium = random.sample(medium_examples, min(10, len(medium_examples)))
sampled_hard = random.sample(hard_examples, min(10, len(hard_examples)))

# Combine all sampled data
sampled_data = sampled_easy + sampled_medium + sampled_hard

# Create lists to hold full_video_url and difficulty
full_video_urls = []
difficulties = []

# Compute full_video_url for each item and collect difficulty
for item in sampled_data:
    full_video_url = os.path.join(item['videos_url'][:-1] + "_merged", f"{item['index']}_E_merged.mp4")
    full_video_urls.append(full_video_url)
    difficulties.append(item['difficulty'])

# Create a DataFrame
df = pd.DataFrame({'full_video_url': full_video_urls, 'difficulty': difficulties})

# Save to CSV without the index
df.to_csv('VAR_Data-v6-11.csv', index=False)

print("CSV file 'sampled_full_video_urls.csv' with 'difficulty' column has been created.")
