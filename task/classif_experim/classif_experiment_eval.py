'''
Example code for evaluation of the classification models on the official test dataset.
'''
from classif_experim.classif_model_builder import (
    build_classif_model_on_full_train,
    load_or_build_classif_fulltrain_model
)
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_utils import binary_labels_to_str, save_text_category_predictions_to_json
from data_tools.dataset_loaders import load_texts_and_ids_from_json
from data_tools.evaluation_utils import evaluate_classif_predictions
from settings import TEST_DATASET_EN, TEST_DATASET_ES


def evaluate_on_test_dataset(model: SklearnTransformerClassif, lang: str, positive_class: str, model_label: str, rseed=35412):
    '''
    :param positive_class: 'conspiracy' or 'critical', depending on how the model was trained
    :return:
    '''
    test_fname = TEST_DATASET_EN if lang == 'en' else TEST_DATASET_ES
    txt, ids = load_texts_and_ids_from_json(test_fname)
    cls_pred = model.predict(txt)
    cls_pred = binary_labels_to_str(cls_pred, positive_class)
    pred_fname = f'predictions_{lang}_{model_label}_rseed[{rseed}].json'
    save_text_category_predictions_to_json(ids, cls_pred, pred_fname)
    # assuming that the test datasets contain the correct class labels, not only the texts and ids
    evaluate_classif_predictions(pred_fname, test_fname, positive_class)

def build_eval_model(lang, positive_class='conspiracy', baseline=False, augment_data=False, rseed=35412):
    if baseline:
        model_name = 'bert-base-cased' if lang == 'en' else 'dccuchile/bert-base-spanish-wwm-cased'
        model_label = 'bert-baseline'
    else:
        model_name = "cardiffnlp/twitter-xlm-roberta-large-2022"
        model_label = "XLMR-large-2022"
        model_lang = 'combined'
    model = load_or_build_classif_fulltrain_model(model_lang, model_name, model_label, augment_data=augment_data,
                                              positive_class=positive_class, rseed=rseed)
    evaluate_on_test_dataset(model, lang, positive_class=positive_class)

if __name__ == '__main__':
    build_eval_model('en')


