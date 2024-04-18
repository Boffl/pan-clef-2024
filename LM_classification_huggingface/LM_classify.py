
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
import argparse
import json
from tqdm import tqdm


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



def classify(text, model, tokenizer, labels=["yes", "no"]):
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
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
                new_inputs = tokenizer([new_text], return_tensors="pt").to(device)
                logits = model(**new_inputs).logits.squeeze()[-1]

    return label_scores




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("datafile")
    parser.add_argument("--prompt_file", help="Prompt to use. Should be a txt file containing placeholders <labels> and <text>", default="simple_prompt.txt")
    parser.add_argument("--labels", help="Labels separated by whitespace. Example: 'yes no'", default="CONSPIRACY, CRITICAL")
    args = parser.parse_args()
    model_name = args.model_name
    data_file = args.datafile
    prompt_file = args.prompt_file
    labels = args.labels
    label_list = labels.split(", ")

    with open(data_file, "r", encoding="utf-8") as injson:
        dataset = json.load(injson)
    
    with open(prompt_file, "r", encoding="utf-8") as infile:
        prompt = infile.read()

    print("downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name)  #  load_in_8bit=True Quantization, not working at the moment, for some reason...
    model.to(device)
    print("done")

    for datapoint in tqdm(dataset):
        text = datapoint['text']
        id = datapoint["id"]

        # filling in the prompt
        prompt = re.sub("<text>", text, prompt)
        prompt = re.sub("<labels>", labels, prompt)

        label_scores = classify(prompt, model, tokenizer, labels=label_list)
        print(prompt)
        print(label_scores)
        break

