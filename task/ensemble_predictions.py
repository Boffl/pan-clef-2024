import os, re, json

model = "XLMR-large-2022"
langs = ["es", "en"]

files = os.listdir()

for lang in langs:
    prediction_filename_regex = rf"predictions_{lang}_{model}_rseed\[\d+\]\.json"
    predictions_from_models_list = []  # list of lists, each list is the set of predictions by a model
    for filename in files:
        if re.match(prediction_filename_regex, filename):
            with open(os.path.join("",filename), "r", encoding="utf-8") as injson:
                predictions_from_models_list.append(json.load(injson))
    
    ensemble_predictions = []
    count_voting_necessary = 0
    for i, first_model_prediction in enumerate(predictions_from_models_list[0]):
        prediction_list = [first_model_prediction["category"]]
        for other_model_predictions in predictions_from_models_list[1:]:
            other_model_prediction = other_model_predictions[i]  # get the other models prediction for the same id
            assert first_model_prediction["id"] == other_model_prediction["id"]
            # append the prediction_list
            prediction_list.append(other_model_prediction["category"])
        
        # majority voting
        voted_label = max(set(prediction_list), key=prediction_list.count)

        # count if the voting was necessary...
        if not all(pred==prediction_list[0] for pred in prediction_list):
            count_voting_necessary += 1
    
        ensemble_predictions.append({
            "id": first_model_prediction["id"],
            "category": voted_label
        })

    with open(f"predictions_{lang}_{model}_ensemble.json", "w", encoding="utf-8") as outjson:
        json.dump(ensemble_predictions, outjson)

    print(f"Voting for {lang} was necessary in {count_voting_necessary} out of {len(predictions_from_models_list[0])} cases")

