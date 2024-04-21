import os
import pandas as pd


def load_pickle_to_df(pickle_path="./data/pickle/"):
  df = pd.read_pickle(pickle_path)
  return df
