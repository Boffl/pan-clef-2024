import json

langs = ["en", "es"]
prompt = "Labeling a text as conspiratorial when it is, in fact, merely oppositional to mainstream views, could potentially lead those who were simply asking questions closer to conspiracy communities. On the other hand, in the context of the COVID-19 Pandemic, identifying conspiratorial content that tries to frame the pandemic or public health decisions as a result of a malevolent conspiracy by secret influential groups is important. Considering this, would you label this as conspiratorial?"

for lang in langs:

    new_dataset = []
    with open(f"../../../data/dataset_{lang}_split_train.json", "r", encoding="utf8") as injson:
        orig_dataset = json.load(injson)
    
    for datapoint in orig_dataset:
        text = datapoint["text"]
        label = datapoint["category"]
        instruction = prompt + "\n" + text
        new_dataset.append({
            'instruction': instruction,
            'input': '',
            'output': label
        })
    
    with open(f"llama_instruct_dataset_train{lang}.json", "w", encoding="utf-8") as outjson:
        json.dump(new_dataset, outjson)