import os
import glob
from task.data_tools.pickle_utils import load_pickle_to_df

pickle_path = "./data/pickle/"
pickle_file = os.path.join(pickle_path, "dataset_ES_translated.pkl")

df = load_pickle_to_df(pickle_file)
print(df.head(10))
