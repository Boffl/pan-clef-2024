# Classifying with a generative LM from Huggingface

The script ```LM_classify.py``` evaluates on the whole dataset that is provided, cross-val to come soon... <br> 
Finetuning also to come... Stay tuned for updates ;)

### Inputs
- positional arguments:
    - <tt>model_name</tt>: Any huggingface LM
    - <tt>datafile</tt>: Filepath to a datafile that has the form of data_en_train.json

- optional arguments:
    - <tt>--prompt_file</tt>: A txt file that is used for the prompt. It should contain placeholders ```<label>```and ```<text>```. Default value is <tt>simple_prompt.txt</tt>
    - <tt>--labels</tt>: Labels separated by comma. Default is <tt>'CONSPIRACY, CRITICAL'</tt> . <br>

Example call:
```
python LM_classify.py sshleifer/tiny-gpt2 ..\..\data\dataset_en_split_dev.json
```

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



