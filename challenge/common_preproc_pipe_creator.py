import re

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class CommonPreProcPipeCreator:
    @classmethod
    def build_pipe(cls, relevant_cols: list):
        pipeline_steps = [('columns selector', FunctionTransformer(lambda df: df[relevant_cols])),
                          ('add time based columns', FunctionTransformer(cls.__add_time_based_cols)),
                          ('add cancellation policy features',
                           FunctionTransformer(cls.__add_cancellation_policy_features))
                          ]

        return Pipeline(pipeline_steps)

    @classmethod
    def __add_time_based_cols(cls, df: pd.DataFrame) -> pd.DataFrame:
        df['stay_length'] = cls.__get_days_between_dates(df.checkout_date, df.checkin_date)
        df['time_registered_pre_book'] = cls.__get_days_between_dates(df.checkin_date, df.hotel_live_date)
        df['booking_to_arrival_time'] = cls.__get_days_between_dates(df.checkin_date, df.booking_datetime)
        df['checkin_week_of_year'] = cls.__get_week_of_year(df.checkin_date)
        df['booking_week_of_year'] = cls.__get_week_of_year(df.booking_datetime)
        df['booked_on_weekend'] = cls.__get_booked_on_weekend(df.booking_datetime)
        df['is_weekend_holiday'] = cls.__get_weekend_holiday(df.checkin_date, df.checkout_date)
        df['is_local_holiday'] = cls.__get_local_holiday(df.origin_country_code, df.hotel_country_code)

        return df

    @classmethod
    def __add_cancellation_policy_features(cls, features: pd.DataFrame) -> pd.DataFrame:
        # TODO - clean up
        cancellation_policy = features.cancellation_policy_code
        features['n_policies'] = cancellation_policy.apply(lambda policy: len(policy.split('_')))
        days_until_policy = cancellation_policy.apply(cls.__get_days_until_policy)

        features['min_policy_days'] = days_until_policy.apply(min)
        features['max_policy_days'] = days_until_policy.apply(max)

        x = features.apply(cls.__get_money_lost_per_policy, axis='columns')
        features['max_policy_cost'], features['min_policy_cost'], features['part_min_policy_cost'] = \
            list(map(list, zip(*x)))

        features['min_policy_cost'] = features['min_policy_cost'].apply(min)
        features['part_min_policy_cost'] = features['part_min_policy_cost'].apply(min)
        features['max_policy_cost'] = features['max_policy_cost'].apply(max)

        return features

    @staticmethod
    def __get_days_between_dates(dates1: pd.Series, dates2: pd.Series):
        return (dates1 - dates2).apply(lambda period: period.days)

    @staticmethod
    def __get_week_of_year(dates):
        return dates.apply(lambda d: d.weekofyear)

    @staticmethod
    def __get_booked_on_weekend(dates):
        return dates.apply(lambda d: d.day_of_week >= 4)

    @staticmethod
    def __get_weekend_holiday(in_date, out_date):
        return list(map(lambda d: (d[1] - d[0]).days <= 3 and d[0].dayofweek >= 4, zip(in_date, out_date)))

    @staticmethod
    def __get_local_holiday(col1, col2):
        return list(map(lambda x: x[0] == x[1], zip(col1, col2)))

    @staticmethod
    def __get_days_until_policy(policy_code: str) -> list:
        policies = policy_code.split('_')
        return [int(policy.split('D')[0]) if 'D' in policy else 0 for policy in policies]

    @staticmethod
    def __get_policy_cost(policy, stay_cost, stay_length, time_until_checkin):
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

    @classmethod
    def __get_money_lost_per_policy(cls, features: pd.Series) -> list:
        policies = features.cancellation_policy_code.split('_')
        stay_cost = features.original_selling_amount
        stay_length = features.stay_length
        time_until_checkin = features.booking_to_arrival_time
        policy_cost = [cls.__get_policy_cost(policy, stay_cost, stay_length, time_until_checkin) for policy in policies]

        return list(map(list, zip(*policy_cost)))


if __name__ == '__main__':
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
    CommonPreProcPipeCreator.build_pipe(RELEVANT_COLUMNS)
