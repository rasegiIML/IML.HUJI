import dataclasses
from typing import NoReturn

import numpy as np
from sklearn.pipeline import Pipeline

from IMLearn import BaseEstimator


class GeneralCancellationEstimator(BaseEstimator):
    def __init__(self):
        raise NotImplementedError

    def add_model(self, X: np.ndarray, y: np.ndarray, preproc_pipe: Pipeline):
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

    def build_pipeline(self):
        raise NotImplementedError
