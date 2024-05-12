import os
import glob
from task.data_tools.pickle_utils import load_pickle_to_df

temp_path = "./data/temp/"

date_dir = [d for d in os.listdir(temp_path) if os.path.isdir(os.path.join(temp_path, d))]
most_recent_directory = max(date_dir, key=lambda d: os.path.getctime(os.path.join(temp_path, d)))
most_recent_directory_path = os.path.join(temp_path, most_recent_directory)

time_dir = [d for d in os.listdir(most_recent_directory_path) if os.path.isdir(os.path.join(most_recent_directory_path, d))]
most_recent_subdirectory = max(time_dir, key=lambda d: os.path.getctime(os.path.join(most_recent_directory_path, d)))
most_recent_subdirectory_path = os.path.join(most_recent_directory_path, most_recent_subdirectory)

# Loop through all pickle files in the most recently created subdirectory
pickle_files = glob.glob(os.path.join(most_recent_subdirectory_path, "*.pkl"))
for pickle_file in pickle_files:
    # Process each pickle file as needed
    print("Processing:", pickle_file)
    df = load_pickle_to_df(pickle_file, pickle_path=most_recent_subdirectory_path)
    print(df.head(10))
