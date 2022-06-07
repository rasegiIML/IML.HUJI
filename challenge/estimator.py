from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, f1_score


class PeriodCancellationEstimator(BaseEstimator):
    def __init__(self) -> PeriodCancellationEstimator:
        super().__init__()
        self._fit_model: RandomForestClassifier = None

    def get_params(self, deep=False):
        return {'threshold': self.thresh}

    def set_params(self, threshold) -> PeriodCancellationEstimator:
        self.thresh = threshold

        return self

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self._fit_model = RandomForestClassifier(random_state=0).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._fit_model.predict(X)

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray):
        RocCurveDisplay.from_estimator(self._fit_model, X, y)

    def score(self, X: pd.DataFrame, y: pd.Series):
        return f1_score(y, self._predict(X), average='macro')
