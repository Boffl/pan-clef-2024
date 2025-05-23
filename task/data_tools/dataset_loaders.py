import json
import random
from typing import List, Tuple

import pandas as pd

from data_tools.dataset_utils import reconstruct_spacy_docs_from_json, BINARY_MAPPING_CONSPIRACY_POS, \
    BINARY_MAPPING_CRITICAL_POS, span_annot_to_spanf1_format, validate_json_annotations, is_empty_annot
from settings import (
    TRAIN_DATASET_EN,
    TRAIN_DATASET_ES,
    TRAIN_DATASET_COMBINED,
    SPLIT_TRAIN_DATASET_EN,
    SPLIT_TRAIN_DATASET_ES,
    SPLIT_DEV_DATASET_EN,
    SPLIT_DEV_DATASET_ES,
    TEST_DATASET_EN,
    AUG_DATASET_EN,
    AUG_DATASET_ES,
    AUG_DATASET_COMBINED
)

def load_aug_dataset_classification(lang, string_labels=False, positive_class='conspiracy'):
    if lang == 'en': fname = AUG_DATASET_EN
    elif lang == 'es': fname = AUG_DATASET_ES
    elif lang == 'combined': fname = AUG_DATASET_COMBINED
    else: raise ValueError(f'Unknown language: {lang}')
    with open(fname, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    # convert to a format suitable for classification
    texts = pd.Series([doc['text'] for doc in dataset])
    if string_labels: classes = pd.Series([doc['category'] for doc in dataset])
    else:
        if positive_class == 'conspiracy': binmap = BINARY_MAPPING_CONSPIRACY_POS
        elif positive_class == 'critical': binmap = BINARY_MAPPING_CRITICAL_POS
        else: raise ValueError(f'Unknown positive class: {positive_class}')
        classes = [binmap[doc['category']] for doc in dataset]
        classes = pd.Series(classes)
    ids = pd.Series([doc['id'] for doc in dataset])
    return texts, classes, ids
    

def load_dataset_full(lang, format='docbin'):
    '''
    Load .json dataset and, optionally, convert it to .docbin format.
    :param format: 'docbin' or 'json'
    :return:
    '''
    print(f'Loading official JSON {lang} dataset')
    if lang == 'en': fname = TRAIN_DATASET_EN
    elif lang == 'es': fname = TRAIN_DATASET_ES
    elif lang == 'combined': fname = TRAIN_DATASET_COMBINED
    else: raise ValueError(f'Unknown language: {lang}')
    if format == 'docbin':
        dataset = reconstruct_spacy_docs_from_json(fname, lang)
    elif format == 'json':
        with open(fname, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
    else: raise ValueError(f'Unknown format: {format}')
    return dataset

def load_dataset_split(lang, format='docbin'):
    '''
    Load .json dataset and, optionally, convert it to .docbin format.
    :param format: 'docbin' or 'json'
    :return:
    '''
    print(f'Loading train and dev JSON {lang} dataset')
    if lang == 'en':
        fname_train = SPLIT_TRAIN_DATASET_EN
        fname_dev = SPLIT_DEV_DATASET_EN
    elif lang == 'es':
        fname_train = SPLIT_TRAIN_DATASET_ES
        fname_dev = SPLIT_DEV_DATASET_ES
    else: raise ValueError(f'Unknown language: {lang}')
    if format == 'docbin':
        dataset_train = reconstruct_spacy_docs_from_json(fname_train, lang)
        dataset_dev = reconstruct_spacy_docs_from_json(fname_dev, lang)
    elif format == 'json':
        with open(fname_train, 'r', encoding='utf-8') as file:
            dataset_train = json.load(file)
        with open(fname_dev, 'r', encoding='utf-8') as file:
            dataset_dev = json.load(file)
    else: raise ValueError(f'Unknown format: {format}')
    return dataset_train, dataset_dev

def load_dataset_classification(lang, string_labels=False, positive_class='conspiracy'):
    '''
    Load official .json dataset and convert it to a format suitable for classification.
    :param lang: 'en', 'es' or 'combined'
    :param string_labels: if True, return orig. string labels from json,
            otherwise return binary labels, 0 for negative, 1 for positive class
    :param positive_class: 'conspiracy' or 'critical'
    :return: three pandas series: texts, binary classes (1 - positive, 0 - negative), text ids
    '''
    dataset = load_dataset_full(lang, format='json')
    # convert to a format suitable for classification
    texts = pd.Series([doc['text'] for doc in dataset])
    if string_labels: classes = pd.Series([doc['category'] for doc in dataset])
    else:
        if positive_class == 'conspiracy': binmap = BINARY_MAPPING_CONSPIRACY_POS
        elif positive_class == 'critical': binmap = BINARY_MAPPING_CRITICAL_POS
        else: raise ValueError(f'Unknown positive class: {positive_class}')
        classes = [binmap[doc['category']] for doc in dataset]
        classes = pd.Series(classes)
    ids = pd.Series([doc['id'] for doc in dataset])
    return texts, classes, ids

def calculate_json_dataset_stats(dset: List, label=''):
    '''
    Calculate and print the following statistics for the dataset:
    number of documents, proportions of the text 'category' classes, proportions of the span annotation classes
    (for each span category, calculate the proportion of documents that have at least one span of that category)
    :param dset: dataset, in the format produced by docbin_to_json(), and loaded by load_official_dataset()
    :return:
    '''
    if label: print(f'STATISTICS FOR {label}')
    num_docs = len(dset)
    text_categ = [doc['category'] for doc in dset]
    span_annot = [doc['annotations'] for doc in dset]
    span_annot = [set([ann['category'] for ann in spans]) for spans in span_annot]
    all_span_categories = set([ann for ann_set in span_annot for ann in ann_set])
    # text category proportions
    text_categ_counts = pd.Series(text_categ).value_counts()
    text_categ_props = text_categ_counts / num_docs
    print(f'Number of documents: {num_docs}')
    print(' ; '.join([f'{categ}: {prop*100:.3f}%' for categ, prop in text_categ_props.items() if categ in ['CRITICAL', 'CONSPIRACY']]))
    # span annotation proportions, for categories in all_span_categories
    span_annot_flags = {ann: [ann in ann_set for ann_set in span_annot] for ann in all_span_categories}
    span_annot_props = {ann: sum(flags)/num_docs for ann, flags in span_annot_flags.items()}
    span_categs = sorted(span_annot_props.items())
    print(' ; '.join([f'{ann}: {prop*100:.3f}%' for ann, prop in span_categs]))
    print()

def load_texts_and_ids_from_json(json_file: str) -> Tuple[List[str], List[str]]:
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = [item['text'] for item in data]
    ids = [item['id'] for item in data]
    return texts, ids

def load_span_annotations_from_json(json_file: str, span_f1_format=True) -> List[List[dict]]:
    '''
    Load span annotations from a .json file.
    :param span_f1_format: if True, the annotations are in the format used for span-F1 score calculation:
            map of document ids to list of annotations, where each annotation is a range of character indices.
            If False, return list of per-document annotations formatted as in the original .json file.
    :return:
    '''
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if span_f1_format:
        result = {}
        for item in data:
            f1annot = []
            annots = item['annotations']
            validate_json_annotations(annots)
            if not is_empty_annot(annots):
                for annot in annots: f1annot.append(span_annot_to_spanf1_format(annot))
            result[item['id']] = f1annot
        return result
    else:
        return [item['annotations'] for item in data]
    
def split_train_set(lang, rndm_seed=42):
    # Load your dataset from JSON file
    with open(f'../data/dataset_{lang}_train.json', 'r') as f:
        dataset = json.load(f)

    # Set seed for reproducibility
    random.seed(rndm_seed)
    # Shuffle the dataset to ensure randomness
    random.shuffle(dataset)

    # Calculate the index to split the dataset
    split_index = int(len(dataset) * 0.9)

    # Split the dataset into training and development sets
    train_set = dataset[:split_index]
    dev_set = dataset[split_index:]

    # Save the training and development sets to JSON files
    with open(f'../data/dataset_{lang}_split_train.json', 'w') as f:
        json.dump(train_set, f, indent=4)
    with open(f'../data/dataset_{lang}_split_dev.json', 'w') as f:
        json.dump(dev_set, f, indent=4)


if __name__ == '__main__':
    #calculate_json_dataset_stats(load_dataset_full('en', format='json'), label='EN')
    load_span_annotations_from_json(TEST_DATASET_EN, span_f1_format=True)


