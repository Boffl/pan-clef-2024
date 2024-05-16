
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re, sys
import torch
import argparse
import json
from datetime import datetime
from tqdm import tqdm
from typing import List
from pathlib import Path

# to be able to access the modules above, even when running with __name__ == __main__
sys.path.append("..")
from data_tools.evaluation_utils import evaluate_classif_predictions
# from classif_experim.classif_utils import classif_scores
# from data_tools.dataset_utils import save_text_category_predictions_to_json

def save_text_category_predictions_to_json(ids: List[str], predictions: List[str], json_file: str):
    data = [{'id': id, 'category': pred} for id, pred in zip(ids, predictions)]
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json_data)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def classify(text, model, tokenizer, labels=["yes", "no"]):

    # whyyyyy does the truncation argument seem to have no effect??? what is wrong???
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**model_inputs).logits.squeeze()[-1]
    label_scores = {}

    for label in labels:
        label_scores[label] = 0
        tokens = tokenizer(label)["input_ids"]  # tokens needed for the label
        if len(tokens)==1:  # We only need one token to represent the label
            label_scores[label] = logits[int(tokens[0])].item()
        else:
            new_inputs = model_inputs  # we'll have to go along here
            for token in tokens:
                label_scores[label] += logits[int(token)].item()
                token_text = tokenizer.decode(token)
                new_text = text + token_text
                new_inputs = tokenizer([new_text], return_tensors="pt", truncation=True).to(device)
                with torch.no_grad():
                    logits = model(**new_inputs).logits.squeeze()[-1]

    # return the key with the highest value
    return max(label_scores.items(), key=lambda x:x[1])[0]



if __name__ == "__main__":
    ########################
    # Example call:
    # !python LM_classify.py distilbert/distilgpt2 ../../data/dataset_en_split_dev.json  --prompt_file prompt1.txt
    #########################
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("data_file")
    parser.add_argument("--prompt_file", help="Prompt to use. Should be a txt file containing placeholders <labels> and <text>", default="simple_prompt.txt")
    parser.add_argument("--labels", help="Labels separated by comma. Example: 'yes, no'", default="CONSPIRACY, CRITICAL")
    args = parser.parse_args()
    model_name = args.model_name
    data_file = args.data_file
    prompt_file = args.prompt_file
    labels = args.labels
    label_list = labels.split(", ")

    with open(data_file, "r", encoding="utf-8") as injson:
        dataset = json.load(injson)
    
    with open(prompt_file, "r", encoding="utf-8") as infile:
        prompt_template = infile.read()

    print("downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    
    # adjust the max length for truncation of the tokenizer, so we have enough space for the labels
    longest_label = max([len(tokenizer.encode(label)) for label in label_list])
    tokenizer.model_max_length -= longest_label
    
    # Quantization, as shown here: https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing#scrollTo=VPD7QS_DR-mw
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")  #  load_in_8bit=True Quantization, not working at the moment, for some reason...
    print("download done")
    ids = []
    predictions = []
    for datapoint in tqdm(dataset):
        text = datapoint['text']
        ids.append(datapoint['id'])

        # Generate response

        classification_done = False
        while not classification_done:
      
            # filling in the prompt
            prompt = re.sub("<text>", text, prompt_template)
            prompt = re.sub("<labels>", labels, prompt)
            try:
                label = classify(prompt, model, tokenizer, labels=label_list)
                classification_done = True
            except torch.cuda.OutOfMemoryError:
                print("The input was to long, cutting it in half...")
                torch.cuda.empty_cache()
                # cut the text in half
                text = " ".join(text.split(" ")[len(text.split())/2:])
            
        predictions.append(label)
        torch.cuda.empty_cache()

    # saving the results, make a sensible filename
    timestamp = datetime.now().strftime('%m-%d_%H_%M')
    # convert filepath to linux (in every case), take out the filename without the .json
    data_file_name = Path(data_file).as_posix().split('/')[-1].split(".")[0]
    outfile_name = f"predictions/{prompt_file.split('.')[0]}_{model_name.split('/')[-1]}_{data_file_name}_{timestamp}.json"
    save_text_category_predictions_to_json(ids, predictions, outfile_name)

    evaluate_classif_predictions(outfile_name, data_file, 'conspiracy')

