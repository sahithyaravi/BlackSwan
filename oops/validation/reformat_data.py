from tqdm import tqdm
import json
import pandas as pd
import argparse 


def get_ground_truths(prev_gt, checkboxes, new_gt):
    # Based on the checkboxes, pick the appropriate ground truths
    ground_truths = []
    types = []
    for i, checkbox in enumerate(checkboxes):
        if checkbox == 'yes':
            ground_truths.append(prev_gt[i])
            gt_type = "valid"
            types.append(gt_type)
        else:
            ground_truths.append(new_gt[i]) if type(new_gt) == list else new_gt
            gt_type = "invalid"
            types.append(gt_type)
    return ground_truths, types





def split_ground_truths(data):
    new_items_task1 = []
    new_items_task2 = []
    new_items_task3a = []
    new_items_task3b = []

    for item in tqdm(data):
        # Task 1
        task1_ground_truths = item['task1_gt']
        for idx, gt in enumerate(task1_ground_truths, start=1):
            if len(gt) >= 3:
                new_item = {
                    'videos_url': item['videos_url'],
                    'preevent': item['preevent'],
                    'reference_text': gt,  # Set the individual ground truth
                    'task_idx': f"task1{idx}"  # Add the task identifier
                }
                new_items_task1.append(new_item)

        # Task 2
        task2_ground_truths, types = get_ground_truths(item['task1_gt'], item['task2_checkboxes'], item['task2_gt'])
        for idx, gt in enumerate(task2_ground_truths, start=1):
            if len(gt) > 3:
                new_item = {
                    'videos_url': item['videos_url'],
                    'preevent': item['preevent'],
                    'postevent': item['postevent'],
                    'reference_text': gt,
                    'task_idx': f"task2{idx}"
                }
                new_items_task2.append(new_item)

        # # Task 3
        task3_ground_truths, types = get_ground_truths(task2_ground_truths, item['task3_checkboxes'], item['task3_gt'])
        for idx, gt in enumerate(task3_ground_truths, start=0):
            if len(gt) > 1:
                gt_type = types[idx]
                if gt_type == "valid":
                    new_item = {
                        'videos_url': item['videos_url'],
                        'preevent': item['preevent'],
                        'postevent': item['postevent'],
                        'event': item['event'],
                        'reference_text': gt,
                        'task_idx': f"task3{idx}"
                    }
                    new_items_task3a.append(new_item)

            print(item['task3_gt'])
            # if gt_type == "invalid":
            new_item = {
                'videos_url': item['videos_url'],
                'preevent': item['preevent'],
                'postevent': item['postevent'],
                'event': item['event'],
                'reference_text': item['task3_gt'],
                'task_idx': f"task3{idx}"
            }
            new_items_task3b.append(new_item)
    df1 = pd.DataFrame(new_items_task1)
    df1.drop_duplicates(["reference_text"], inplace=True)
    df1.to_csv("annotation_input_task1.csv", index=False)

    df2 = pd.DataFrame(new_items_task2)
    df2.drop_duplicates(["reference_text"], inplace=True)
    df2.to_csv("annotation_input_task2.csv", index=False)

    df3 = pd.DataFrame(new_items_task3a)
    df3.drop_duplicates(["reference_text"], inplace=True)
    df3.to_csv("annotation_input_task3a.csv", index=False)

    df4 = pd.DataFrame(new_items_task3b)
    df4.drop_duplicates(["reference_text"], inplace=True)
    df4.to_csv("annotation_input_task3b.csv", index=False)

    print(len(df1), len(df2), len(df3), len(df4))


def main(input_path, output_path):
    # Load your JSON data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Process the tasks and evaluate responses
    processed_data = split_ground_truths(data)

    
    # Output or save the metrics as needed


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Process a JSON into the format expected by template.")
    
    # Define input and output arguments
    parser.add_argument('--input_path', type=str, 
                        help="Path to the input JSON file.", 
                        default="/h/sahiravi/VAR/data/VAR_Data_v9p2_caption.json", required=False)
    parser.add_argument('--output_path', type=str, help="Path to save the output JSON file.", required=False, 
                        default="annotation_input_task1.csv")
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_path, args.output_path)