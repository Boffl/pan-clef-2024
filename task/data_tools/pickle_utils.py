import os
import pandas as pd


def load_pickle_to_df(pickle_name, pickle_path="./data/pickle/"):
  full_path = os.path.join(pickle_path, pickle_name)
  df = pd.read_pickle(full_path)
  return df
