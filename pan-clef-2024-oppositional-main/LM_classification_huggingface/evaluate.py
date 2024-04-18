import sys
sys.path.append("..")
from data_tools.evaluation_utils import evaluate_classif_predictions


pred_file = sys.argv[1]
gold_file = sys.argv[2]

evaluate_classif_predictions(pred_file, gold_file, positive_class="conspiracy")