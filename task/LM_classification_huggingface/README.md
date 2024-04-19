# Classifying with a generative LM from Huggingface

The script ```LM_classify.py``` evaluates on the whole dataset that is provided, cross-val to come soon... <br> 
Finetuning also to come... Stay tuned for updates ;)

### Inputs
it takes as positional arguments:
- A model name, that is any huggingface LM
- A datafile, that is in the form of data_en_train.json
And as optional arguments:
- <tt>--prompt_file</tt>: a txt file that is used for the prompt. It should contain placeholders ```<label>```and ```<text>```. Default value is <tt>simple_prompt.txt</tt>
- <tt>--labels</tt>: Labels separated by comma. Default is <tt>'CONSPIRACY, CRITICAL'</tt> .

### Outputs
The script will create a file in the ```/predictions``` directory, where the answers of the model are saved in the appropriate json format. The filename contains the information about the experiment:
```
<model-name>_<dataset-name>_<date_time(mm-dd_HH-MM)>.json
```
Based on this file the performance is calculated and printed to the console. <br>

#### Evaluation Script
Since the results are saved, you can also recalculate performance of past experiments by calling ```evaluate.py``` and giving it the predictions and gold labels as arguments. Example call:
```
evaluate.py tiny-gpt2_dataset_en_split_dev_04-18_19_10.json dataset_en_split_dev.json 
```



