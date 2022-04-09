import dataclasses
from collections import namedtuple
from copy import deepcopy, copy
from typing import NoReturn

import numpy as np
import pandas as pd
from numpy import datetime64
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from IMLearn import BaseEstimator
from challenge.common_preproc_pipe_creator import CommonPreProcPipeCreator
from challenge.period_cancellation_estimator import PeriodCancellationEstimator

Period = namedtuple("Period", ("days_until", 'length'))


def get_response_for_period(train_data: pd.DataFrame, period: Period) -> pd.Series:
    return (train_data.cancellation_datetime >=
            train_data.booking_datetime + pd.DateOffset(day=period.days_until)) & \
           (train_data.cancellation_datetime <=
            train_data.booking_datetime + pd.DateOffset(day=period.days_until + period.length))


class ModelForPeriodExistsError(BaseException):
    pass


class GeneralCancellationEstimator:
    def __init__(self, period_length: int, cancellation_period_start=None):
        super().__init__()
        self._models = {}
        self.cancellation_period_start = cancellation_period_start
        self.__def_days_until_cancellation = 0
        self.__period_length = period_length

    def _get_days_until_cancellation_period(self, data_row: pd.Series):
        return (self.cancellation_period_start - data_row.booking_datetime).days \
            if self.cancellation_period_start is not None else self.__def_days_until_cancellation

    def add_model(self, X: np.ndarray, y: np.ndarray, pipe: Pipeline, period: Period, threshold=0.5):
        if period.days_until in self._models:
            raise ModelForPeriodExistsError(f'There already exists a model with {period.days_until} '
                                            f'days until the start of the relevant cancellation period.')

        assert period.length == self.__period_length, \
            f'Error: estimator only deals with periods of length {self.__period_length}.'

        train_X = pipe.transform(X)

        model_estimator = PeriodCancellationEstimator(threshold).fit(train_X, y)
        pipe.steps.append(('estimator', model_estimator))

        self._models[period.days_until] = pipe

    def predict(self, X: pd.DataFrame) -> pd.Series:
        periods = X.apply(self._get_days_until_cancellation_period, axis='columns')

        return X.groupby(periods, as_index=False) \
            .apply(lambda data: pd.Series(self._models[data.name].predict(data), index=data.index)) \
            .droplevel(0, axis='index').sort_index()

    def test_models(self, test_data: pd.DataFrame):
        for days_until_canc_period_start in self._models:
            self.__def_days_until_cancellation = days_until_canc_period_start
            test_data_resp = get_response_for_period(test_data,
                                                     Period(days_until_canc_period_start, self.__period_length))

            model_score = self._models[days_until_canc_period_start].score(test_data, test_data_resp)

            print(f'Score for model with {days_until_canc_period_start} days until start'
                  f' of cancellation period: {model_score:.3f}')

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
    __COL_TYPE_CONVERSIONS = {'checkout_date': 'datetime64',
                              'checkin_date': 'datetime64',
                              'hotel_live_date': 'datetime64',
                              'booking_datetime': 'datetime64'}

    def build_pipeline(self, train_data: pd.DataFrame,
                       cancellation_period_start: np.datetime64) -> GeneralCancellationEstimator:
        base_pipe = self.__create_common_preproc_pipeline()

        general_estimator = GeneralCancellationEstimator(self.period_length, cancellation_period_start)

        for days_until_cancellation_period in range(self.min_days_until_cancellation_period,
                                                    self.max_days_until_cancellation_period + 1):
            print(f'Creating model for {days_until_cancellation_period} days until cancellation.')

            preproc_pipe = deepcopy(base_pipe)
            period = Period(days_until_cancellation_period, self.period_length)
            train_data['cancelled_in_period'] = get_response_for_period(train_data.astype(self.__COL_TYPE_CONVERSIONS),
                                                                        period)

            preproc_pipe = self.__add_period_dependent_preproc_to_pipe(preproc_pipe, train_data)

            general_estimator.add_model(train_data.drop('cancelled_in_period', axis='columns'),
                                        train_data.cancelled_in_period, preproc_pipe, period)

        return general_estimator

    @classmethod
    def __create_common_preproc_pipeline(cls) -> Pipeline:
        return CommonPreProcPipeCreator.build_pipe(cls.__RELEVANT_COLUMNS)

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
    def __create_col_prob_mapper(col: str, mapper: dict, MISSING_FILL=0.5):
        mapper = copy(mapper)

        def map_col_to_prob(df):
            df[col] = df[col].apply(lambda x: mapper.get(x, MISSING_FILL))

            return df

        return map_col_to_prob
