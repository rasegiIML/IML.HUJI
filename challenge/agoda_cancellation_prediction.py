from matplotlib import pyplot as plt

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd

__DEBUG = True


def get_days_between_dates(dates1: pd.Series, dates2: pd.Series):
    return (dates1 - dates2).apply(lambda period: period.days)


def process_categorical_data(features: pd.DataFrame, cat_vars: list, one_hot=False, calc_probs=True) -> pd.DataFrame:
    assert one_hot ^ calc_probs, \
        'Error: can only do either one-hot encoding or probability calculations, not neither/both!'
    # one-hot encoding
    if one_hot:
        features = pd.get_dummies(features, columns=cat_vars)

    # category probability preprocessing - make each category have its success percentage
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
    # TODO - add original_selling_amount column once the forum question is answered
    NONE_OUTPUT_COLUMNS = ['checkin_date',
                           'checkout_date',
                           'booking_datetime',
                           'hotel_live_date']
    CATEGORICAL_COLUMNS = ['hotel_star_rating',
                           'guest_nationality_country_name',
                           'charge_option',
                           'accommadation_type_name',
                           'language',
                           'is_first_booking',
                           'customer_nationality',
                           'original_payment_currency',
                           'is_user_logged_in']
    RELEVANT_COLUMNS = ['no_of_adults',
                        'no_of_children',
                        'no_of_extra_bed',
                        'no_of_room'] + NONE_OUTPUT_COLUMNS + CATEGORICAL_COLUMNS
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

    features = process_categorical_data(features, CATEGORICAL_COLUMNS)

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


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    id1, id2, id3 = 209855253, 205843964, 212107536
    evaluate_and_export(estimator, test_X, f"{id1}_{id2}_{id3}.csv")

    # plot results
    estimator.plot_roc_curve(test_X, test_y)

    plt.xlim(0)
    plt.ylim(0)

    plt.show()
