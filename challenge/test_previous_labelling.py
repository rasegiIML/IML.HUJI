import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    previous_label_path = '209855253_205843964_212107536.csv'
    groundtruth_path = '../datasets/test_set_week_1_labels.csv'

    previous_labels = pd.read_csv(previous_label_path).predicted_values
    groundtruth = pd.read_csv(groundtruth_path, sep='|').label

    print(f'Accuracy for previous labelling: {accuracy_score(previous_labels, groundtruth):.3f}')
