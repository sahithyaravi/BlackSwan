import json
import pandas as pd
import requests
from openai import AzureOpenAI
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from io import BytesIO
import random

# Configuration and Constants
from keys import GPT4_KEY, GPT4_ENDPOINT

# Define the OpenAI client
class OpenAIClient:
    def __init__(self, api_key, endpoint):
        self.client = AzureOpenAI(
            api_version="2024-02-15-preview",
            azure_endpoint=endpoint,
            api_key=api_key
        )
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

    def get_response(self, messages):
        response = self.client.chat.completions.create(
            model="gpt4o",
            messages=messages,
            max_tokens=500,
        )
        return response

# Define utility functions
class Utils:
    @staticmethod
    def extract_answer(string_input):
        try:
            answer = string_input.split("Answer:")[1]
        except IndexError:
            answer = string_input
        return answer.strip()

    @staticmethod
    def display_images(img_urls):
        images = []
        for url in img_urls:
            response = requests.get(url)
            img_data = BytesIO(response.content)
            img_pil = Image.open(img_data)
            images.append(img_pil)

        num_images = len(images)
        fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2.5, 3))
        
        if num_images == 1:
            axs = [axs]

        for i, img in enumerate(images):
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f'Image {i + 1}')
        
        plt.show()

# Define the VideoQAProcessor class
class VideoQAProcessor:
    def __init__(self, client, dataset_path, output_file, num_frames):
        self.client = client
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.num_frames = num_frames
        self.dataset = self.load_dataset()

    def load_dataset(self):
        with open(self.dataset_path, 'r') as file:
            return json.load(file)

    def extract_frames(self, video_frames):
        frames = []
        for i in range(1, self.num_frames + 1):
            url = f"{video_frames}/frame_{i}.jpg"
            frames.append(url)
        return frames

    def mcq_task1(self, task1_frames, task2_frames, options):
        task1_prompt = f"""Given the above frames from the beginning and end of a video. Which of the following could have happened in the middle?
        {options}
        
        Return your response as:
        Reason: A reason for choosing the answer in maximum of 10 words
        Answer: Your choice e.g., 'A'"""

        messages = [{"role": "user", "content": [{"type": "text", "text": task1_prompt}]}]
        messages.append({"role": "user", "content": [{"type": "text", "text": "Here are frames from the beginning of a video."}]})

        for frame in task1_frames:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})

        messages.append({"role": "user", "content": [{"type": "text", "text": "Here are frames from the end of a video."}]})

        for frame in task2_frames:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})

        return messages, self.client.get_response(messages)

    def mcq_task2(self, task_frames, options):
        task2_prompt = f"""Given the above frames from the video, choose which of the following options hold true based on the video context?
        {options}
        
        Return your response as:
        Reason: A reason for choosing the answer in maximum of 10 words
        Answer: Your choice e.g., 'A'"""

        messages = [{"role": "user", "content": [{"type": "text", "text": task2_prompt}]}]
        messages.append({"role": "user", "content": [{"type": "text", "text": "Here are the frames of the video."}]})

        for frame in task_frames:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": frame}}]})

        return messages, self.client.get_response(messages)

    def process_questions(self):
        final_responses = []
        correct_task1 = correct_task2 = total_task1 = total_task2 = 0
        correct_overall = total_overall = 0
        failed_task1 = failed_task2 = 0
        df = pd.DataFrame()
        random_sample = random.sample(self.dataset, 200)

        for item in tqdm(random_sample):
            try:
                preevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_A_preevent")
                event_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_B_event")
                postevent_frames = self.extract_frames(item['frames_url'] + f"{item['index']}_C_postevent")

                all_frames = preevent_frames + event_frames + postevent_frames

                mcq_type = str(item["mcq_task"])
                options = item["mcq_options"]
                answer_index = item["mcq_label"]

                multiple_choice_string = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(options)])

                print(multiple_choice_string)

                if mcq_type == "1":
                    _, task_responses = self.mcq_task1(preevent_frames, postevent_frames, multiple_choice_string)
                    task = "Task 1"
                else:
                    _, task_responses = self.mcq_task2(all_frames, multiple_choice_string)
                    task = "Task 2"

                fr = item
                fr['gpt4_answer'] = Utils.extract_answer(task_responses.choices[0].message.content)
                fr['gpt4_reason'] = task_responses.choices[0].message.content
                fr["preevent_frames"] = preevent_frames
                fr["postevent_frames"] = postevent_frames
                final_responses.append(fr)

                gpt4_answer = fr['gpt4_answer']
                is_correct = (gpt4_answer.startswith('A') and answer_index == 0) or (gpt4_answer.startswith('B') and answer_index == 1) or (gpt4_answer.startswith('C') and answer_index == 2)

                if task == "Task 1":
                    correct_task1 += is_correct
                    total_task1 += 1
                    if not is_correct:
                        failed_task1 += 1
                else:
                    correct_task2 += is_correct
                    total_task2 += 1
                    if not is_correct:
                        failed_task2 += 1

                correct_overall += is_correct
                total_overall += 1

                print(f"Task 1 Running Accuracy: {correct_task1 / total_task1 if total_task1 else 0:.2f}")
                print(f"Task 2 Running Accuracy: {correct_task2 / total_task2 if total_task2 else 0:.2f}")
                print(f"Overall Running Accuracy: {correct_overall / total_overall:.2f}")

                row_df = pd.DataFrame([fr])
                df = pd.concat([df, row_df], ignore_index=True)
                df.to_csv(self.output_file, index=False)

            except Exception as e:
                if mcq_type == "1":
                    failed_task1 += 1
                else:
                    failed_task2 += 1
                print(f"Error processing item {item['index']}: {e}")

        print(f"Task 1 Final Accuracy: {correct_task1 / total_task1 if total_task1 else 0:.2f}")
        print(f"Task 1 Final Failures: {failed_task1 / total_task1 if total_task1 else 0:.2f}")
        print(f"Task 2 Final Accuracy: {correct_task2 / total_task2 if total_task2 else 0:.2f}")
        print(f"Task 2 Final Failures: {failed_task2 / total_task2 if total_task2 else 0:.2f}")
        print(f"Overall Final Accuracy: {correct_overall / total_overall:.2f}")
        return final_responses


# Main script execution
if __name__ == "__main__":
    client = OpenAIClient(api_key=GPT4_KEY, endpoint=GPT4_ENDPOINT)
    processor = VideoQAProcessor(
        client=client,
        dataset_path="mcq_list_v9-11_v2_gpt.json",
        output_file="mcq_gpt4_preview.csv",
        num_frames=10
    )
    output = processor.process_questions()

    with open("output_mcq.json", 'w') as f:
        json.dump(output, f, indent=4, separators=(',', ':'))
