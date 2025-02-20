import json
import sys

def combine_responses(file1, file2, file3, output_file):
    # Load the data from each JSON file
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    with open(file3, 'r') as f:
        data3 = json.load(f)

    # Ensure the length of data is the same in each file
    if not (len(data1) == len(data2) == len(data3)):
        print("Error: JSON files contain different numbers of entries.")
        sys.exit(1)

    combined_data = []

    # Combine responses for each task
    for entry1, entry2, entry3 in zip(data1, data2, data3):
        # Copy all fields from entry1 (assumes each entry in all three files has the same structure)
        combined_entry = entry1.copy()
        
        # Create combined response lists for each task
        combined_entry['task1_responses'] = [
            entry1.get('task1_response', ''),
            entry2.get('task1_response', ''),
            entry3.get('task1_response', '')
        ]
        combined_entry['task2_responses'] = [
            entry1.get('task2_response', ''),
            entry2.get('task2_response', ''),
            entry3.get('task2_response', '')
        ]
        combined_entry['task3_responses'] = [
            entry1.get('task3_response', ''),
            entry2.get('task3_response', ''),
            entry3.get('task3_response', '')
        ]

        # Remove the individual task response fields from the combined entry
        combined_entry.pop('task1_response', None)
        combined_entry.pop('task2_response', None)
        combined_entry.pop('task3_response', None)

        # Append the modified entry to the combined data list
        combined_data.append(combined_entry)

    # Save the combined data to the output file
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Combined JSON saved to {output_file}")

# Example usage
combine_responses('/h/sahiravi/VAR/results/VAR_Data_vila_7B_gen1.json', 
                  '/h/sahiravi/VAR/results/VAR_Data_vila_7B_gen2.json',
                   '/h/sahiravi/VAR/results/VAR_Data_vila_7B_gen2.json' ,
                   '/h/sahiravi/VAR/results/VAR_Data_.json' )
