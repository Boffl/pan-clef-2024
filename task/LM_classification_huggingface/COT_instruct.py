from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse
import re, json, sys
from tqdm import tqdm
from typing import List
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import jsonlines
from multiprocessing import Pool

sys.path.append("..")
from data_tools.evaluation_utils import evaluate_classif_predictions

def save_text_category_predictions_to_json(ids: List[str], predictions: List[str], responses, json_file: str):
    data = [{'id': id, 'category': pred, 'response':response} for id, pred, response in zip(ids, predictions, responses)]
    json_data = json.dumps(data, ensure_ascii=False, indent=2)
    with open(json_file, 'w', encoding='utf-8') as file:
        file.write(json_data)

def classify(messages, model, tokenizer, labels=["yes", "no"]):

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits.squeeze()[-1]
    label_scores = {}

    for label in labels:
        label_scores[label] = 0
        tokens = tokenizer(label)["input_ids"]  # tokens needed for the label
        if len(tokens)==1:  # We only need one token to represent the label
            label_scores[label] = logits[int(tokens[0])].item()
        else:
            # reset for the next label
            new_inputs = input_ids  
            new_messages = messages
            new_messages.append({
                "role": "assistant", "content": ""  # to include the response
            })
            for token in tokens:
                label_scores[label] += logits[int(token)].item()
                token_text = tokenizer.decode(token)
                new_messages[-1]["content"] += token_text
                new_inputs = tokenizer.apply_chat_template(
                                    new_messages,
                                    add_generation_prompt=True,
                                    return_tensors="pt"
                                ).to(model.device)
                with torch.no_grad():
                    logits = model(new_inputs).logits.squeeze()[-1]

    # return the key with the highest value
    return max(label_scores.items(), key=lambda x:x[1])[0]

def generate (messages, model, tokenizer):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)



if __name__ == "__main__":

    #####################
    # Example call
    # python COT_instruct.py meta-llama/Meta-Llama-3-8B-Instruct ../../data/dataset_en_split_toy.json instruct_prompt2.jsonl
    ######################
    
    print("new_stuff")
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("data_file")
    parser.add_argument("prompt_file")
    args = parser.parse_args()
    model_id = args.model_name
    data_file = args.data_file
    prompt_file = args.prompt_file


    with open(data_file, "r", encoding="utf-8") as injson:
        dataset = json.load(injson)

    labels = ["yes", "no"]
    label_dict = {
        "yes": "CONSPIRACY",
        "no": "CRITICAL"
    }


    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Quantization, as shown here: https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing#scrollTo=VPD7QS_DR-mw
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")  
    print("download done")



    prompt_messages = []
    with jsonlines.open(prompt_file) as reader:
        for line in reader:
            prompt_messages.append(line)

    ids = []
    predictions = []
    responses = []
    for datapoint in tqdm(dataset):
        text = datapoint['text']
        ids.append(datapoint['id'])

        # filling in the prompt, make a new one every time...
        messages = deepcopy(prompt_messages)
        messages[1]["content"] = re.sub(r"<text>", text, messages[1]["content"])
        
        # Generating the first prompt, reasoning :D
        response_text = generate(messages, model, tokenizer)
        responses.append(response_text)

        # adding prompt for the classification
        messages += [
            {"role": "assistant", "content": response_text},
            {"role": "user", "content": f"Given your analysis would you consider this text conspiratorial? Answer with one word yes/no."}
        ]
        # generate a response to classify the input
        response_text = generate(messages, model, tokenizer)
        
        # if the response (.lower().strip()) isnot in the classes use the classify function
        if response_text.lower().strip() not in labels:
            response_text = classify(messages, model, tokenizer, labels)
        
        prediction = label_dict[response_text.lower().strip()]
        predictions.append(prediction)
        torch.cuda.empty_cache()

        # saving the results, make a sensible filename
    timestamp = datetime.now().strftime('%m-%d_%H_%M')
    # convert filepath to linux (in every case), take out the filename without the .json
    data_file_name = Path(data_file).as_posix().split('/')[-1].split(".")[0]
    outfile_name = f"predictions/instruct_prompt_{model_id.split('/')[-1]}_{data_file_name}_{timestamp}.json"
    save_text_category_predictions_to_json(ids, predictions, responses, outfile_name)

    evaluate_classif_predictions(outfile_name, data_file, 'conspiracy')


