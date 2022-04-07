import dataclasses
from collections import namedtuple
from copy import deepcopy
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from IMLearn import BaseEstimator
from challenge.common_preproc_pipe_creator import CommonPreProcPipeCreator

Period = namedtuple("Period", ("days_until", 'length'))


class GeneralCancellationEstimator(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.__models = {}

    def add_model(self, X: np.ndarray, y: np.ndarray, preproc_pipe: Pipeline, period: Period):
        raise NotImplementedError

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        raise NotImplementedError

    def _predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


@dataclasses.dataclass
class GeneralCancellationEstimatorBuilder:
    period_length: int
    min_days_until_checkin: int
    max_days_until_checkin: int

    __NONE_OUTPUT_COLUMNS = ['checkin_date',
                             'checkout_date',
                             'booking_datetime',
                             'hotel_live_date',
                             'hotel_country_code',
                             'origin_country_code',
                             'cancellation_policy_code']
    __CATEGORICAL_COLUMNS = ['hotel_star_rating',
                             'guest_nationality_country_name',
                             'charge_option',
                             'accommadation_type_name',
                             'language',
                             'is_first_booking',
                             'customer_nationality',
                             'original_payment_currency',
                             'is_user_logged_in',
                             ]
    __RELEVANT_COLUMNS = ['no_of_adults',
                          'no_of_children',
                          'no_of_extra_bed',
                          'no_of_room',
                          'original_selling_amount'] + __NONE_OUTPUT_COLUMNS + __CATEGORICAL_COLUMNS
    __DEF_PATH = "../datasets/agoda_cancellation_train.csv"
    __COL_TYPE_CONVERSIONS = {'checkout_date': 'datetime64',
                              'checkin_date': 'datetime64',
                              'hotel_live_date': 'datetime64',
                              'booking_datetime': 'datetime64'}

    def build_pipeline(self, train_path: str == __DEF_PATH) -> GeneralCancellationEstimator:
        base_pipe = self.__create_common_preproc_pipeline()

        train_data = self.__read_data_file(train_path)

        general_estimator = GeneralCancellationEstimator()

        for days_until_checkin in range(self.max_days_until_checkin, self.max_days_until_checkin + 1):
            # raise NotImplementedError
            preproc_pipe = deepcopy(base_pipe)
            period = Period(days_until_checkin, self.period_length)
            train_data['cancelled_in_period'] = self.__get_response_for_period(train_data, period)

            preproc_pipe = self.__add_period_dependent_preproc_to_pipe(preproc_pipe, train_data)

            general_estimator.add_model(train_data.drop('cancelled_in_period', axis='columns'),
                                        train_data.cancelled_in_period, preproc_pipe, period)

        return general_estimator

    @classmethod
    def __read_data_file(cls, path: str) -> pd.DataFrame:
        return pd.read_csv(path).drop_duplicates() \
            .astype(cls.__COL_TYPE_CONVERSIONS)

    @classmethod
    def __create_common_preproc_pipeline(cls) -> Pipeline:
        return CommonPreProcPipeCreator.build_pipe(cls.__RELEVANT_COLUMNS)
