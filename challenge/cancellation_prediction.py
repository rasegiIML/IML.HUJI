import re
from collections import namedtuple
from copy import copy
from datetime import datetime
from typing import NoReturn

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from IMLearn import BaseEstimator
from challenge.general_cancellation_estimator import GeneralCancellationEstimatorBuilder
from challenge.period_cancellation_estimator import PeriodCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

__DEBUG = False

Period = namedtuple("Period", ("days_until", 'length'))


def read_data_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path).drop_duplicates() \
        .astype({'checkout_date': 'datetime64',
                 'checkin_date': 'datetime64',
                 'hotel_live_date': 'datetime64',
                 'booking_datetime': 'datetime64'})


def get_days_between_dates(dates1: pd.Series, dates2: pd.Series):
    return (dates1 - dates2).apply(lambda period: period.days)


def create_col_prob_mapper(col: str, mapper: dict):
    mapper = copy(mapper)

    def map_col_to_prob(df):
        df[col] = df[col].apply(mapper.get)

        return df

    return map_col_to_prob


def add_categorical_prep_to_pipe(train_features: pd.DataFrame, pipeline: Pipeline, cat_vars: list, one_hot=False,
                                 calc_probs=True) -> Pipeline:
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
                                   FunctionTransformer(create_col_prob_mapper(cat_var, map_cat_to_prob))))

    return pipeline


def get_week_of_year(dates):
    return dates.apply(lambda d: d.weekofyear)


def get_booked_on_weekend(dates):
    return dates.apply(lambda d: d.day_of_week >= 4)


def get_weekend_holiday(in_date, out_date):
    return list(map(lambda d: (d[1] - d[0]).days <= 3 and d[0].dayofweek >= 4, zip(in_date, out_date)))


def get_local_holiday(col1, col2):
    return list(map(lambda x: x[0] == x[1], zip(col1, col2)))


def get_days_until_policy(policy_code: str) -> list:
    policies = policy_code.split('_')
    return [int(policy.split('D')[0]) if 'D' in policy else 0 for policy in policies]


def get_policy_cost(policy, stay_cost, stay_length, time_until_checkin):
    """
    returns tuple of the format (max lost, min lost, part min lost)
    """
    if policy == 'UNKNOWN':
        return 0, 0, 0
    nums = tuple(map(int, re.split('[a-zA-Z]', policy)[:-1]))
    if 'D' not in policy:  # no show is suppressed
        return 0, 0, 0
    if 'N' in policy:
        nights_cost = stay_cost / stay_length * nums[0]
        min_cost = nights_cost if time_until_checkin <= nums[1] else 0
        return nights_cost, min_cost, min_cost / stay_cost
    elif 'P' in policy:
        nights_cost = stay_cost * nums[0] / 100
        min_cost = nights_cost if time_until_checkin <= nums[1] else 0
        return nights_cost, min_cost, min_cost / stay_cost
    else:
        raise Exception("Invalid Input")


def get_money_lost_per_policy(features: pd.Series) -> list:
    policies = features.cancellation_policy_code.split('_')
    stay_cost = features.original_selling_amount
    stay_length = features.stay_length
    time_until_checkin = features.booking_to_arrival_time
    policy_cost = [get_policy_cost(policy, stay_cost, stay_length, time_until_checkin) for policy in policies]

    return list(map(list, zip(*policy_cost)))


def add_cancellation_policy_features(features: pd.DataFrame) -> pd.DataFrame:
    cancellation_policy = features.cancellation_policy_code
    features['n_policies'] = cancellation_policy.apply(lambda policy: len(policy.split('_')))
    days_until_policy = cancellation_policy.apply(get_days_until_policy)

    features['min_policy_days'] = days_until_policy.apply(min)
    features['max_policy_days'] = days_until_policy.apply(max)

    x = features.apply(get_money_lost_per_policy, axis='columns')
    features['max_policy_cost'], features['min_policy_cost'], features['part_min_policy_cost'] = \
        list(map(list, zip(*x)))

    features['min_policy_cost'] = features['min_policy_cost'].apply(min)
    features['part_min_policy_cost'] = features['part_min_policy_cost'].apply(min)
    features['max_policy_cost'] = features['max_policy_cost'].apply(max)

    return features


def add_time_based_cols(df: pd.DataFrame) -> pd.DataFrame:
    df['stay_length'] = get_days_between_dates(df.checkout_date, df.checkin_date)
    df['time_registered_pre_book'] = get_days_between_dates(df.checkin_date, df.hotel_live_date)
    df['booking_to_arrival_time'] = get_days_between_dates(df.checkin_date, df.booking_datetime)
    df['checkin_week_of_year'] = get_week_of_year(df.checkin_date)
    df['booking_week_of_year'] = get_week_of_year(df.booking_datetime)
    df['booked_on_weekend'] = get_booked_on_weekend(df.booking_datetime)
    df['is_weekend_holiday'] = get_weekend_holiday(df.checkin_date, df.checkout_date)
    df['is_local_holiday'] = get_local_holiday(df.origin_country_code, df.hotel_country_code)

    return df


def create_pipeline_from_data(filename: str, period: Period):
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
    features = read_data_file(filename)

    features['cancelled_in_period'] = ((features.cancellation_datetime >=
                                        features.checkin_date + pd.DateOffset(day=period.days_until)) & \
                                       (features.cancellation_datetime <=
                                        features.checkin_date + pd.DateOffset(day=period.days_until + period.length))) \
        .astype(int)

    pipeline_steps = [('columns selector', FunctionTransformer(lambda df: df[RELEVANT_COLUMNS])),
                      ('add time based columns', FunctionTransformer(add_time_based_cols)),
                      ('add cancellation policy features', FunctionTransformer(add_cancellation_policy_features))
                      ]

    pipeline = Pipeline(pipeline_steps)
    pipeline = add_categorical_prep_to_pipe(features, pipeline, CATEGORICAL_COLUMNS)

    pipeline.steps.append(('drop irrelevant columns',
                           FunctionTransformer(lambda df: df.drop(NONE_OUTPUT_COLUMNS, axis='columns'))))

    return features.drop('cancelled_in_period', axis='columns'), features.cancelled_in_period, pipeline


def evaluate_and_export(estimator: BaseEstimator, X: pd.DataFrame, filename: str):
    preds = (~estimator.predict(X)).astype(int)
    pd.DataFrame(preds, columns=["predicted_values"]).to_csv(filename, index=False)


def create_estimator_for_period_from_data(period: Period, path="../datasets/agoda_cancellation_train.csv",
                                          threshold: float = 0.5,
                                          optimize_threshold=False, frac: float = 0.75, debug=False) -> Pipeline:
    np.random.seed(0)

    # Load data
    raw_df, cancellation_labels, pipeline = create_pipeline_from_data(path, period)

    train_X, train_y, test_X, test_y = split_train_test(raw_df, cancellation_labels, train_proportion=frac)
    processed_test_X = pipeline.transform(test_X)
    processed_train_X = pipeline.transform(train_X)

    # Fit model over data
    estimator = PeriodCancellationEstimator(threshold).fit(processed_train_X, train_y)
    pipeline.steps.append(('estimator', estimator))

    if optimize_threshold:
        unopt_est = PeriodCancellationEstimator()
        param_optim = GridSearchCV(unopt_est, {'threshold': np.linspace(0, 1)})
        param_optim.fit(train_X, train_y)
        print(f'Optimal params: {param_optim.best_params_}')
        estimator.thresh = param_optim.best_params_['threshold']

    # plot results
    if debug:
        estimator.plot_roc_curve(processed_test_X, test_y)

        plt.xlim(0)
        plt.ylim(0)

        plt.show()

    print(f'Accuracy score on test data: {estimator.score(processed_test_X, test_y)}')

    return pipeline


def export_test_data(pipeline: Pipeline, path="../datasets/test_set_week_1.csv") -> NoReturn:
    data = read_data_file(path)

    # Store model predictions over test set
    id1, id2, id3 = 209855253, 205843964, 212107536
    evaluate_and_export(pipeline, data, f"{id1}_{id2}_{id3}.csv")


if __name__ == '__main__':
    temp_period = Period(5, 7)
    pipeline = GeneralCancellationEstimatorBuilder(PERIOD_LENGTH, min_days_until_checkin, max_days_until_checkin) \
        .build_pipeline()
    pipeline = create_estimator_for_period_from_data(temp_period)
    # export_test_data(pipeline)