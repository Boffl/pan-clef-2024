# silly things to make it run on Nadia's silly python setup, pls ignore
# import sys
# sys.path.append('C:/Users/Nadia Timoleon/Documents/GitHub/pan-clef-2024/task')

# # Some setup for Hilal python setup.
# import pathlib
# import sys
# # Look for python code 1 directory up.
# sys.path.append(pathlib.Path(__file__).parent.parent.resolve().as_posix())
# import os
# # Set active directory 2 directories up, to find ./data folder with pickles.
# os.chdir(pathlib.Path(__file__).parent.parent.parent.resolve().as_posix())

import tqdm
import argparse
import pandas as pd
import os

from datetime import datetime

from task.data_tools.pickle_utils import load_pickle_to_df

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

lang_dicts = {
    "EN": {
        "source_lang": "EN",
        "target_lang": "ES",
        "data_file": "./data/pickle/dataset_en_train.pkl",
        "model_name": "Helsinki-NLP/opus-mt-en-es"
    },
    "ES": {
        "source_lang": "ES",
        "target_lang": "EN",
        "data_file": "./data/pickle/dataset_es_train.pkl",
        "model_name": "Helsinki-NLP/opus-mt-es-en"
        # Alternative larger model with 1024 token capacity
        #"model_name": "mrm8488/mbart-large-finetuned-opus-es-en-translation"
    }
}

def load_and_filter_data(data_file, max_tokens=512):
    df = load_pickle_to_df(data_file)
    # Filter out rows more tokens than the model can handle
    df = df[df["text"].apply(lambda x: len(x.split()) <= max_tokens)]
    return df


def translate_text(df, model_name):
    # Load the model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
    translations, categories = [], []
    skipped = 0
    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        text, category = row["text"], row["category"]
        try:
            input_ids = tokenizer.encode(text, return_tensors="pt")
            outputs = model.generate(input_ids)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Skipping row {idx}: {e}")
            skipped += 1
            continue
        translations.append(translated_text)
        categories.append(category)
    
    # create a new dataframe with the translated text
    df_translations = pd.DataFrame({"text": translations, "category": categories})
    # rename the index to 'id'
    df_translations.index.name = "id"
    print(f"Number of skipped rows: {skipped}")

    return df_translations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Translate Spanish to English")
    parser.add_argument("--source_lang",choices=["EN", "ES"],help="Choose the source language to translate from: 'EN' or 'ES'")
    parser.add_argument("--test", default="FALSE",choices=["TRUE", "FALSE"],help="Run the script on a small subset of the data for testing purposes. Default is 'FALSE.")
    parser.add_argument("--test_size",type=float,default=0.01,help="Fraction of the original dataset to be translated. Default is 0.01.")
    args = parser.parse_args()

    source_lang = args.source_lang
    test = args.test
    test_size = args.test_size

    output_folder = "./data/"

    # Get the language specific parameters
    lang_dict = lang_dicts[source_lang]
    source_lang = lang_dict["source_lang"]
    target_lang = lang_dict["target_lang"]
    data_file = lang_dict["data_file"]
    model_name = lang_dict["model_name"]

    print("Loading data...")
    df = load_and_filter_data(data_file) 

    if test == "TRUE":
        print("Running a test translation on a small subset of the data.")
        print("The translation will be saved to the temp folder")
        df = df.sample(frac=test_size, random_state=42)

        # make directory with the current datetime as name
        current_time = datetime.now().strftime("%Hh%Mm%Ss")   
        output_folder += f"temp/{datetime.now().date()}/{current_time}/"
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder += "pickle/"

    print(f"Data loaded.\nNumber of sequences in the dataset: {len(df)}")
    print(f"Translating text from {source_lang} to {target_lang}...")
    df_translations = translate_text(df, model_name)
    # Save the translated data to a pickle file
    df_translations.to_pickle(output_folder + f"dataset_{target_lang}_translated.pkl")
