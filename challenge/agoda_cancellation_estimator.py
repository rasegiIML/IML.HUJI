from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, accuracy_score

from IMLearn.base import BaseEstimator


class AgodaCancellationEstimator(BaseEstimator):
    def __init__(self, threshold: float = None) -> AgodaCancellationEstimator:
        super().__init__()
        self.__fit_model: RandomForestClassifier = None
        self.thresh = threshold

    def get_params(self, deep=False):
        return {'threshold': self.thresh}

    def set_params(self, threshold) -> AgodaCancellationEstimator:
        self.thresh = threshold

        return self

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self.__fit_model = RandomForestClassifier(random_state=0).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.__fit_model.predict_proba(X)[:, 1]
        return probs > self.thresh if self.thresh is not None else probs

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pass

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray):
        RocCurveDisplay.from_estimator(self.__fit_model, X, y)

    def score(self, X: pd.DataFrame, y: pd.Series):
        return accuracy_score(y, self._predict(X))
