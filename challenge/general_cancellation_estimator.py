import dataclasses
from collections import namedtuple
from copy import deepcopy, copy
from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from IMLearn import BaseEstimator
from challenge.common_preproc_pipe_creator import CommonPreProcPipeCreator
from challenge.period_cancellation_estimator import PeriodCancellationEstimator

Period = namedtuple("Period", ("days_until", 'length'))


class ModelForPeriodExistsError(BaseException):
    pass


class GeneralCancellationEstimator:
    def __init__(self):
        super().__init__()
        self.__models = {}

    def __get_period(self, data_row: pd.Series):
        return (data_row.checkin_date - data_row.booking_datetime).days

    def add_model(self, X: np.ndarray, y: np.ndarray, pipe: Pipeline, period: Period, threshold=0.5):
        if period.days_until in self.__models:
            raise ModelForPeriodExistsError(f'There already exists a model with {period.days_until} '
                                            f'days until checkin.')

        train_X = pipe.transform(X)

        model_estimator = PeriodCancellationEstimator(threshold).fit(train_X, y)
        pipe.steps.append(('estimator', model_estimator))

        self.__models[period.days_until] = pipe

    def predict(self, X: pd.DataFrame) -> pd.Series:
        periods = X.apply(self.__get_period, axis='columns')

        return X.groupby(periods, as_index=False).apply(lambda data: self.__models[data.name].predict(data))

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError


@dataclasses.dataclass
class GeneralCancellationEstimatorBuilder:
    period_length: int
    min_days_until_cancellation_period: int
    max_days_until_cancellation_period: int

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

    def build_pipeline(self, train_data: pd.DataFrame) -> GeneralCancellationEstimator:
        base_pipe = self.__create_common_preproc_pipeline()

        general_estimator = GeneralCancellationEstimator()

        for days_until_cancellation_period in range(self.min_days_until_cancellation_period,
                                                    self.max_days_until_cancellation_period + 1):
            preproc_pipe = deepcopy(base_pipe)
            period = Period(days_until_cancellation_period, self.period_length)
            train_data['cancelled_in_period'] = self.__get_response_for_period(train_data, period)

            preproc_pipe = self.__add_period_dependent_preproc_to_pipe(preproc_pipe, train_data)

            general_estimator.add_model(train_data.drop('cancelled_in_period', axis='columns'),
                                        train_data.cancelled_in_period, preproc_pipe, period)

        return general_estimator

    @classmethod
    def __create_common_preproc_pipeline(cls) -> Pipeline:
        return CommonPreProcPipeCreator.build_pipe(cls.__RELEVANT_COLUMNS)

    @staticmethod
    def __get_response_for_period(train_data: pd.DataFrame, period: Period) -> pd.Series:
        return (train_data.cancellation_datetime >=
                train_data.checkin_date + pd.DateOffset(day=period.days_until)) & \
               (train_data.cancellation_datetime <=
                train_data.checkin_date + pd.DateOffset(day=period.days_until + period.length))

    @classmethod
    def __add_period_dependent_preproc_to_pipe(cls, preproc_pipe: Pipeline, train_data: pd.DataFrame) -> Pipeline:
        preproc_pipe = cls.__add_categorical_prep_to_pipe(train_data, preproc_pipe, cls.__CATEGORICAL_COLUMNS)

        preproc_pipe.steps.append(('drop irrelevant columns',
                                   FunctionTransformer(lambda df: df.drop(cls.__NONE_OUTPUT_COLUMNS, axis='columns'))))

        return preproc_pipe

    @classmethod
    def __add_categorical_prep_to_pipe(cls, train_features: pd.DataFrame, pipeline: Pipeline, cat_vars: list,
                                       one_hot=False, calc_probs=True) -> Pipeline:
        assert one_hot ^ calc_probs, \
            'Error: can only do either one-hot encoding or probability calculations, not neither/both!'
        # one-hot encoding
        if one_hot:
            # TODO - use sklearn OneHotEncoder
            pipeline.steps.append(('one-hot encoding',
                                   FunctionTransformer(lambda df: pd.get_dummies(df, columns=cat_vars))))

        # category probability preprocessing - make each category have its success percentage
        if calc_probs:
            for cat_var in cat_vars:
                map_cat_to_prob: dict = train_features.groupby(cat_var, dropna=False) \
                    .cancelled_in_period.mean().to_dict()

                pipeline.steps.append((f'map {cat_var} to prob',
                                       FunctionTransformer(cls.__create_col_prob_mapper(cat_var, map_cat_to_prob))))

        return pipeline

    @staticmethod
    def __create_col_prob_mapper(col: str, mapper: dict):
        mapper = copy(mapper)

        def map_col_to_prob(df):
            df[col] = df[col].apply(mapper.get)

            return df

        return map_col_to_prob
