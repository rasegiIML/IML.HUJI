import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == '__main__':
    previous_label_path = '209855253_205843964_212107536.csv'
    groundtruth_path = '../datasets/test_set_week_1_labels.csv'

    previous_labels = pd.read_csv(previous_label_path).predicted_values
    groundtruth = pd.read_csv(groundtruth_path, sep='|').label

    print(f'Score for previous labelling:\n'
          f'Precision:{precision_score(groundtruth, previous_labels, average="macro"):.3f}\n'
          f'Recall:   {recall_score(groundtruth, previous_labels, average="macro"):.3f}\n'
          f'F1 Macro: {f1_score(groundtruth, previous_labels, average="macro"):.3f}\n'
          f'Accuracy: {accuracy_score(groundtruth, previous_labels):.3f}')
