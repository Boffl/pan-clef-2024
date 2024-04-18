# silly thins to make it run on Nadia's silly python setup, pls ignore
# import sys
# sys.path.append('C:/Users/Nadia Timoleon/Documents/GitHub/pan-clef-2024/task')

import tqdm
import argparse
import pandas as pd

from data_tools.pickle_utils import load_pickle_to_df

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_and_filter_data(data_name):
    df = load_pickle_to_df(data_name)
    
    # only keep sequences with length up to 2000
    df = df[df['text'].apply(lambda x: len(x) <= 2000)]
    return df


def translate_text(df):
    # Load the model and tokenizer
    model_name = "Helsinki-NLP/opus-mt-es-en"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translations = []
    for text in tqdm.tqdm(df['text']):
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model.generate(input_ids)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translations.append(translated_text)
    
    # create a new dataframe with the translated text
    df_translations = pd.DataFrame({"text": translations})
    # rename the index to 'id'
    df_translations.index.name = "id"

    return df_translations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Translate Spanish to English")
    parser.add_argument("source_lang", choices=["en", "es"], help="Choose the source language to translate from: 'en' or 'es'")
    args = parser.parse_args()

    source_lang = args.source_lang

    if source_lang == "en":
        target_lang = "es"
        data_file = "dataset_en_train.pkl"
        model_name = "Helsinki-NLP/opus-mt-en-es"
    elif source_lang == "es":
        target_lang = "en"
        data_name = "dataset_es_train.pkl"
        model_name = "Helsinki-NLP/opus-mt-es-en"
    else:
        raise ValueError("Invalid language, choose 'en' or 'es'")

    print("Loading and filtering data...")
    df = load_and_filter_data(data_name)
    # uncomment the following line to test the code with a smaller dataset
    # df = df.loc[:5].copy()
    print(f"Data loaded and filtered.\nNumber of sequences in the dataset: {len(df)}")
    print(f"Translating text from {source_lang} to {target_lang}...")
    df_translations = translate_text(df)
    # Save the translated data to a pickle file
    df_translations.to_pickle(f"./data/pickle/dataset_{target_lang}_backtranslated.pkl")
