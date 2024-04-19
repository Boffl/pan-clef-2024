import os
import pandas as pd


def load_pickle_to_df(data_name):
  pickle_path = "./data/pickle/"
  data_path = os.path.join(pickle_path, data_name)
  df = pd.read_pickle(data_path)
  return df
