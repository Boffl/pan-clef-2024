import os
import pandas as pd
import json

def json_to_dataframe(json_file):
    with open(json_file, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def convert_json_files_to_pickle(json_dir, pickle_dir):
    # Create pickle directory if it doesn't exist
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir)
    
    # Loop through each json file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            df = json_to_dataframe(json_path)
            df = df[['id', 'text', 'category']].copy()
            pickle_path = os.path.join(pickle_dir, os.path.splitext(filename)[0] + '.pkl')
            df.to_pickle(pickle_path)
            print(f"Converted {filename} to {pickle_path}")

# Example usage
if __name__ == "__main__":
    # Specify the directory containing JSON files and the directory to save pickle files
    json_directory = "data/"
    pickle_directory = "data/pickle/"
    
    # Convert JSON files to pickle
    convert_json_files_to_pickle(json_directory, pickle_directory)