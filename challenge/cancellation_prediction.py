from collections import namedtuple
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from IMLearn import BaseEstimator
from IMLearn.utils import split_train_test
from challenge.general_cancellation_estimator import GeneralCancellationEstimatorBuilder

pd.options.mode.chained_assignment = None  # default='warn'

__DEBUG = False
__DEF_PATH = "../datasets/agoda_cancellation_train.csv"

Period = namedtuple("Period", ("days_until", 'length'))


def read_data_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path).drop_duplicates() \
        .astype({'checkout_date': 'datetime64',
                 'checkin_date': 'datetime64',
                 'hotel_live_date': 'datetime64',
                 'booking_datetime': 'datetime64'})


def evaluate_and_export(estimator: BaseEstimator, X: pd.DataFrame, filename: str):
    preds = estimator.predict(X).astype(int)
    pd.DataFrame(preds, columns=["predicted_values"]).to_csv(filename, index=False)


def export_test_data(pipeline: Pipeline, path="../datasets/test_set_week_1.csv") -> NoReturn:
    data = read_data_file(path)

    # Store model predictions over test set
    id1, id2, id3 = 209855253, 205843964, 212107536
    evaluate_and_export(pipeline, data, f"{id1}_{id2}_{id3}.csv")


if __name__ == '__main__':
    PERIOD_LENGTH = 7
    MIN_DAYS_UNTIL_CANCELLATION_PERIOD = 6
    MAX_DAYS_UNTIL_CANCELLATION_PERIOD = 35
    CANCEL_PERIOD_START = np.datetime64('2018-12-07')

    data = read_data_file(__DEF_PATH)

    train, _, test, _ = split_train_test(data, data.cancellation_datetime, train_proportion=1.)

    general_estimator = GeneralCancellationEstimatorBuilder(PERIOD_LENGTH, MIN_DAYS_UNTIL_CANCELLATION_PERIOD,
                                                            MAX_DAYS_UNTIL_CANCELLATION_PERIOD) \
        .build_pipeline(train, CANCEL_PERIOD_START)

    # general_estimator.test_models(test)

    export_test_data(general_estimator)
