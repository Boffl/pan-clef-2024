# Data augmentation

## Description
The initial task data is limited to 4000 samples per language. We increase the number of samples by translating one language into the other.  

We utilize the ES-EN and EN-ES translation models from Helsinki-NLP, developed for the purposes of the Tatoeba challenge and trained on the opus dataset.  

The models are available at the following links:
- [opus MT ES-EN](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
- [opus MT EN-ES](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)

## Usage
Simple run the `translate.py` script by adding the source language as an argument.
You can also perform a test run by setting the `--test` flag to `True`. This will save the resulting translations in the `temp` data folder.  
e.g. for Spanish to English translation test:

```
cd path/to/project/repo
python -m task.data_augmentation.translate --source_lang ES --test TRUE
```