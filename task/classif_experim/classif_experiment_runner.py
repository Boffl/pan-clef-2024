from copy import copy
import logging
import time
import datetime
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import random

from classif_experim.classif_utils import classif_scores
from classif_experim.hf_skelarn_wrapper import SklearnTransformerClassif
from data_tools.dataset_loaders import load_dataset_classification, load_aug_dataset_classification


def build_transformer_model(model_label, model_hparams, rnd_seed):
    ''' Factory method for building a sklearn-wrapped transformer model.'''
    return SklearnTransformerClassif(hf_model_label=model_label, **model_hparams, rnd_seed=rnd_seed)


def run_classif_crossvalid(lang, model_label, model_params, positive_class='critical', num_folds=5,
                           rnd_seed=None, test=False, pause_after_fold=0, augment_data=False):
    '''
    Run x-fold crossvalidation for a given model, and report the results.
    '''
    logger.info(f'RUNNING crossvalid. for model: {model_label}, augment_data={augment_data}')
    if rnd_seed is None:
        rnd_seed = random.randint(0, 100000)
    score_fns = classif_scores('all')
    texts, classes, txt_ids = load_dataset_classification(lang, positive_class=positive_class)
    if test: texts, classes, txt_ids = texts[:test], classes[:test], txt_ids[:test]
    foldgen = StratifiedKFold(n_splits=num_folds, random_state=rnd_seed, shuffle=True)
    fold_index = 0
    results_df = pd.DataFrame(columns=score_fns.keys())
    conf_mx = None; rseed = rnd_seed
    pred_res = {} # map text_id -> class prediction
    for train_index, test_index in foldgen.split(texts, classes):
        logger.info(f'Starting Fold {fold_index+1}')
        model = build_transformer_model(model_label, model_params, rseed)
        logger.info(f'model built')
        # split data
        txt_tr, txt_tst = texts[train_index], texts[test_index]
        cls_tr, cls_tst = classes[train_index], classes[test_index]
        # logger.info("len of train txt and cls, before augmentation:", len(txt_tr), len(cls_tr))
        id_tst = txt_ids[test_index]
        if augment_data:
            aug_texts, aug_classes, aug_txt_ids = load_aug_dataset_classification(lang, positive_class=positive_class)
            txt_tr = pd.concat([txt_tr, aug_texts], ignore_index=True)
            cls_tr = pd.concat([cls_tr, aug_classes], ignore_index=True)
            logger.info(f"Running on augmented data. Length of the Training set: {len(txt_tr)}, {len(cls_tr)}")
        # train model
        model.fit(txt_tr, cls_tr)
        # evaluate model
        cls_pred = model.predict(txt_tst)
        for txt_id, pred in zip(id_tst, cls_pred):
            assert txt_id not in pred_res
            pred_res[txt_id] = pred
        del model # clear memory
        scores = pd.DataFrame({fname: [f(cls_tst, cls_pred)] for fname, f in score_fns.items()})
        # log scores
        logger.info(f'Fold {fold_index+1} scores:')
        logger.info("; ".join([f"{fname:10}: {f(cls_tst, cls_pred):.3f}" for fname, f in score_fns.items()]))
        # formatted_values = [f"{col:10}: {scores[col].iloc[0]:.3f}" for col in scores.columns]
        results_df = pd.concat([results_df, scores], ignore_index=True)
        conf_mx_tmp = confusion_matrix(cls_tst, cls_pred)
        if conf_mx is None: conf_mx = conf_mx_tmp
        else: conf_mx += conf_mx_tmp
        if pause_after_fold and fold_index < num_folds - 1:
            logger.info(f'Pausing for {pause_after_fold} minutes...')
            time.sleep(pause_after_fold * 60)
        rseed += 1; fold_index += 1
    conf_mx = conf_mx.astype('float64')
    conf_mx /= num_folds
    logger.info('CROSSVALIDATION results:')
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: ' + '; '.join(f'{nm}: {val:.3f}' for nm, val in results_df[fname].describe().items()))
    logger.info('Per-fold scores:')
    # for each score function, log all the per-fold results
    for fname in score_fns.keys():
        logger.info(f'{fname:10}: [{", ".join(f"{val:.3f}" for val in results_df[fname])}]')
    logger.info('Confusion matrix:')
    for r in conf_mx:
        logger.info(', '.join(f'{v:7.2f}' for v in r))
    assert set(pred_res.keys()) == set(txt_ids)
    return pred_res

MAX_SEQ_LENGTH = 256

HF_MODEL_LIST = {
    'en': [
        #'bert-base-cased',  # Baseline provided 
        "Twitter/twhin-bert-large",  # multilingual, trained on twitter
    ],
    'es': [
        #'dccuchile/bert-base-spanish-wwm-cased',  # Baseline
        "bertin-project/bertin-roberta-base-spanish", 
        "Twitter/twhin-bert-large",  # multilingual, trained on twitter
    ],
    'combined': [
        "cardiffnlp/twitter-xlm-roberta-base",  # multilingual, trained on twitter
        "cardiffnlp/twitter-xlm-roberta-large-2022",  # multilingual, trained on twitter, large
    ],
}

# default reasonable parameters for SklearnTransformerBase
HF_CORE_HPARAMS = {
    'learning_rate': 2e-5,
    'num_train_epochs': 3,
    'warmup': 0.1,
    'weight_decay': 0.01,
    'batch_size': 16,  # batch size 32 gives worse results and is not much faster, so leave it there
}

DEFAULT_RND_SEED = 564671

logger = None
def setup_logging(log_filename):
    global logger
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        handlers=[
            logging.FileHandler(log_filename),  # Log to a file with timestamp in its name
            logging.StreamHandler()  # Log to console
        ],
        force=True  #Note in Colab I had to add force=True, for the logger to show the info messages If it causes problems, take it out
    )
    logger = logging.getLogger('')

def run_classif_experiments(lang, num_folds, rnd_seed, test=False, experim_label=None,
                            pause_after_fold=0, pause_after_model=0, max_seq_length=MAX_SEQ_LENGTH,
                            positive_class='critical', model_list=None, augment_data=False):
    '''
    :param positive_class: 'critical' or 'conspiracy'
    :return:
    '''
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experim_label = f'{experim_label}_rseed_{rnd_seed}' if experim_label else f'rseed_{rnd_seed}'
    log_filename = f"classification_experiments_{experim_label}_{timestamp}.log"
    setup_logging(log_filename)
    models = HF_MODEL_LIST[lang] if model_list is None else model_list
    params = copy(HF_CORE_HPARAMS)
    params['lang'] = lang
    params['eval'] = 0.1
    params['max_seq_length'] = max_seq_length
    logger.info(f'RUNNING classif. experiments: lang={lang.upper()}, num_folds={num_folds}, '
                f'max_seq_len={max_seq_length}, eval={params["eval"]}, rnd_seed={rnd_seed}, test={test}')
    logger.info(f'... HPARAMS = {"; ".join(f"{param}: {val}" for param, val in HF_CORE_HPARAMS.items())}')
    init_batch_size = params['batch_size']
    pred_res = {}
    for model in models:
        try_batch_size = init_batch_size
        grad_accum_steps = 1
        while try_batch_size >= 1:
            try:
                params['batch_size'] = try_batch_size
                params['gradient_accumulation_steps'] = grad_accum_steps
                res = run_classif_crossvalid(lang=lang, model_label=model, model_params=params, num_folds=num_folds,
                                             rnd_seed=rnd_seed, test=test, pause_after_fold=pause_after_fold,
                                             positive_class=positive_class, augment_data=augment_data)
                pred_res[model] = res
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logging.warning(
                        f"GPU out of memory using batch size {try_batch_size}. Halving batch size and doubling gradient accumulation steps.")
                    try_batch_size //= 2
                    grad_accum_steps *= 2
                else:
                    raise e
            if try_batch_size < 1:
                logging.error("Minimum batch size reached and still encountering memory errors. Exiting.")
                break
        if pause_after_model:
            logger.info(f'Pausing for {pause_after_model} minutes...')
            time.sleep(pause_after_model * 60)
    return pred_res

def run_all_critic_conspi(seed=DEFAULT_RND_SEED, langs=['en', 'es'], augment_data=False, num_folds=5):
    for lang in langs:
        run_classif_experiments(lang=lang, num_folds=num_folds, rnd_seed=seed, test=None,
                                positive_class='critical', pause_after_fold=1,
                                pause_after_model=2, augment_data=augment_data)

if __name__ == '__main__':
    run_all_critic_conspi(augment_data=True)

