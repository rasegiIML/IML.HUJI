from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, f1_score

from challenge.common_preproc_pipe_creator import CommonPreProcPipeCreator


class PeriodCancellationEstimator(BaseEstimator):
    def __init__(self) -> PeriodCancellationEstimator:
        super().__init__()
        self._fit_model: RandomForestClassifier = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self._fit_model = RandomForestClassifier(random_state=0).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._fit_model.predict(X)

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray):
        RocCurveDisplay.from_estimator(self._fit_model, X, y)

    def score(self, X: pd.DataFrame, y: pd.Series):
        return f1_score(y, self._predict(X), average='macro')


if __name__ == '__main__':
    data = pd.read_csv('cached_data.csv')
    labels = pd.read_csv('cached_labels.csv')

    NONE_OUTPUT_COLUMNS = ['checkin_date',
                           'checkout_date',
                           'booking_datetime',
                           'hotel_live_date',
                           'hotel_country_code',
                           'origin_country_code',
                           'cancellation_policy_code']
    CATEGORICAL_COLUMNS = ['hotel_star_rating',
                           'guest_nationality_country_name',
                           'charge_option',
                           'accommadation_type_name',
                           'language',
                           'is_first_booking',
                           'customer_nationality',
                           'original_payment_currency',
                           'is_user_logged_in',
                           ]
    RELEVANT_COLUMNS = ['no_of_adults',
                        'no_of_children',
                        'no_of_extra_bed',
                        'no_of_room',
                        'original_selling_amount'] + NONE_OUTPUT_COLUMNS + CATEGORICAL_COLUMNS
    pipe = CommonPreProcPipeCreator.build_pipe(RELEVANT_COLUMNS)

    pipe.steps.append(('estimator', PeriodCancellationEstimator()))

    pipe.fit(data, labels)

    predictions = pipe.predict(data)
