import re
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd

# from challenge.agoda_cancellation_estimator_gideoni import AgodaCancellationEstimator

__DEBUG = False


def get_days_between_dates(dates1: pd.Series, dates2: pd.Series):
    return (dates1 - dates2).apply(lambda period: period.days)


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
            map_cat_to_prob: dict = train_features.groupby(cat_var, dropna=False).labels.mean().to_dict()

            def map_cat_col(df: pd.DataFrame) -> pd.DataFrame:
                df[cat_var] = df[cat_var].apply(map_cat_to_prob.get)

                return df

            pipeline.steps.append((f'map {cat_var} to prob', FunctionTransformer(map_cat_col)))

    return pipeline


def process_categorical_data(features: pd.DataFrame, cat_vars: list, one_hot=False, calc_probs=True) -> pd.DataFrame:
    assert one_hot ^ calc_probs, \
        'Error: can only do either one-hot encoding or probability calculations, not neither/both!'
    # one-hot encoding
    if one_hot:
        features = pd.get_dummies(features, columns=cat_vars)

    # category probability preprocessing - make each category have its success percentage
    # TODO - return these dictionaries and add them to the pipeline
    if calc_probs:
        for cat_var in cat_vars:
            map_cat_to_prob: dict = features.groupby(cat_var, dropna=False).labels.mean().to_dict()

            features[cat_var] = features[cat_var].apply(map_cat_to_prob.get)

            if __DEBUG:
                plt.bar(*zip(*map_cat_to_prob.items()))
                plt.title(f'{cat_var} no cancellation probability distribution')
                while not plt.waitforbuttonpress(5):
                    continue

                plt.close()

    return features


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
    # TODO - clean this up and make it work
    cancellation_policy = features.cancellation_policy_code
    features['n_policies'] = cancellation_policy.apply(lambda policy: len(policy.split('_')))
    days_until_policy = cancellation_policy.apply(get_days_until_policy)
    x = features.apply(get_money_lost_per_policy, axis='columns')
    features['max_policy_cost'], features['min_policy_cost'], features['part_min_policy_cost'] = list(
        map(list, zip(*x)))

    features['min_policy_days'] = days_until_policy.apply(min)
    features['max_policy_days'] = days_until_policy.apply(max)

    features['min_policy_cost'] = features['min_policy_cost'].apply(min)
    features['part_min_policy_cost'] = features['part_min_policy_cost'].apply(min)
    features['max_policy_cost'] = features['max_policy_cost'].apply(max)

    return features


def create_pipeline_from_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """

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
    features = pd.read_csv(filename).drop_duplicates() \
        .astype({'checkout_date': 'datetime64',
                 'checkin_date': 'datetime64',
                 'hotel_live_date': 'datetime64',
                 'booking_datetime': 'datetime64'})

    pipeline_steps = [('columns selector', FunctionTransformer(lambda df: df[RELEVANT_COLUMNS]))]

    features['labels'] = features["cancellation_datetime"].isna()

    # preprocessing
    # TODO - move this into a separate function once it becomes too messy

    # feature addition

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

    pipeline_steps.append(('add time based columns', FunctionTransformer(add_time_based_cols)))

    # TODO : check if customer is from holiday country
    pipeline = Pipeline(pipeline_steps)
    pipeline = add_categorical_prep_to_pipe(features, pipeline, CATEGORICAL_COLUMNS)

    pipeline.steps.append(('add cancellation policy features', FunctionTransformer(add_cancellation_policy_features)))

    pipeline.steps.append(('drop irrelevant columns',
                           FunctionTransformer(lambda df: df.drop(NONE_OUTPUT_COLUMNS, axis='columns'))))

    return features.drop('labels', axis='columns'), features.labels, pipeline


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
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
    full_data = pd.read_csv(filename).drop_duplicates() \
        .astype({'checkout_date': 'datetime64',
                 'checkin_date': 'datetime64',
                 'hotel_live_date': 'datetime64',
                 'booking_datetime': 'datetime64'})
    features = full_data[RELEVANT_COLUMNS]
    features['labels'] = full_data["cancellation_datetime"].isna()

    # preprocessing
    # TODO - move this into a separate function once it becomes too messy

    # feature addition
    features['stay_length'] = get_days_between_dates(features.checkout_date, features.checkin_date)
    features['time_registered_pre_book'] = get_days_between_dates(features.checkin_date, features.hotel_live_date)
    features['booking_to_arrival_time'] = get_days_between_dates(features.checkin_date, features.booking_datetime)
    features['checkin_week_of_year'] = get_week_of_year(features.checkin_date)
    features['booking_week_of_year'] = get_week_of_year(features.booking_datetime)
    features['booked_on_weekend'] = get_booked_on_weekend(features.booking_datetime)
    features['is_weekend_holiday'] = get_weekend_holiday(features.checkin_date, features.checkout_date)
    features['is_local_holiday'] = get_local_holiday(features.origin_country_code, features.hotel_country_code)

    # TODO : check if customer is from holiday country
    features = process_categorical_data(features, CATEGORICAL_COLUMNS)
    features = add_cancellation_policy_features(features)

    return features.drop(NONE_OUTPUT_COLUMNS + ['labels'], axis='columns'), features.labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def novel_main():
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    raw_df, _, prep_pipe = create_pipeline_from_data("../datasets/agoda_cancellation_train.csv")
    pipe_res = prep_pipe.transform(raw_df)

    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    id1, id2, id3 = 209855253, 205843964, 212107536
    evaluate_and_export(estimator, test_X, f"{id1}_{id2}_{id3}.csv")

    # plot results
    estimator.plot_roc_curve(test_X, test_y)
    print(f'Accuracy score: {estimator.score(test_X, test_y)}')

    plt.xlim(0)
    plt.ylim(0)

    plt.show()


if __name__ == '__main__':
    novel_main()
