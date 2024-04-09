import json
import sys, os

infilename = sys.argv[1]  # json file
split = sys.argv[2]  # train or dev-set or whole dataset?
lang = sys.argv[3]  # en/es

with open(infilename, "r", encoding="utf-8") as injson:
    data_dict = json.load(injson)

mapping = {
    "CRITICAL": 0,
    "CONSPIRACY": 1
}

labels = []
texts = []

for datapoint in data_dict:
    texts.append(datapoint['text'])
    labels.append(datapoint['category'])
    

with open(os.path.join("raw_data", lang, f"{split}_labels.txt"), "w", encoding="utf-8") as outfile:
    for label in labels:
        outfile.write(f"{mapping[label]}\n")

with open(os.path.join("raw_data", lang, f"{split}_text.txt"), "w", encoding="utf-8") as outfile:
    for text in texts:
        outfile.write(f"{text}\n")