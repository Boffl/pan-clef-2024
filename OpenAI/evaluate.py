from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import sys

gold_file = sys.argv[1]
pred_file = sys.argv[2]

def open_label_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        labels = [int(label) for label in f]
    return labels

gold = open_label_file(gold_file)
pred = open_label_file(pred_file)

print(f"P: {precision_score(gold, pred)}")
print(f"R: {recall_score(gold, pred)}")
print(f"F1: {f1_score(gold, pred)}")
print(f"MCC: {matthews_corrcoef(gold, pred)}")