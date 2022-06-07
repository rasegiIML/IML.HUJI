import os

import pandas as pd

if __name__ == '__main__':
    WEEKS_TO_IGNORE = [9]

    datasets = []
    datasets_folder = 'data'
    for dataset_name in os.listdir(datasets_folder):
        if any(str(week) in dataset_name for week in WEEKS_TO_IGNORE):
            continue

        datasets.append(pd.read_csv(os.path.join(datasets_folder, dataset_name)))

    datasets = pd.concat(datasets)
    datasets.to_csv('cached_data.csv', index=False)

    labels = []
    labels_folder = 'labels'
    for label_name in os.listdir(labels_folder):
        if any(str(week) in label_name for week in WEEKS_TO_IGNORE):
            continue

        labels.append(pd.read_csv(os.path.join(labels_folder, label_name)))

    labels = pd.concat(labels)
    labels.to_csv('cached_labels.csv', index=False)
