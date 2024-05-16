'''
definitions of important setting variables (resources etc.)
create a local settings.py (git ignored) with these variables and their appropriate values
when introducing new variables, add them to this file with empty or default values and commit
'''

# DATASETS
# Original
TRAIN_DATASET_EN = '../data/dataset_en_train.json'
TRAIN_DATASET_ES = '../data/dataset_es_train.json'

# Combined EN and ES
TRAIN_DATASET_COMBINED = '../data/dataset_combined_train.json'

# Train-dev split
SPLIT_TRAIN_DATASET_EN = '../data/dataset_en_split_train.json'
SPLIT_TRAIN_DATASET_ES = '../data/dataset_es_split_train.json'

SPLIT_DEV_DATASET_EN = '../data/dataset_en_split_dev.json'
SPLIT_DEV_DATASET_ES = '../data/dataset_es_split_dev.json'


TEST_DATASET_EN = '../data/dataset_en_test.json'
TEST_DATASET_ES = '../data/dataset_es_test.json'

# Augmented datasets
AUG_DATASET_EN = '../data/dataset_EN_translated.json'
AUG_DATASET_ES = '../data/dataset_ES_translated.json'
AUG_DATASET_COMBINED = '../data/dataset_combined_translated.json'