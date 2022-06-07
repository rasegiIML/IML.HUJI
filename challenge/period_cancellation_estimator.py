from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from IMLearn.base import BaseEstimator


class PeriodCancellationEstimator(BaseEstimator):
    def __init__(self, threshold: float = 0.5) -> PeriodCancellationEstimator:
        super().__init__()
        self._fit_model: RandomForestClassifier = None
        self.thresh = threshold

    def get_params(self, deep=False):
        return {'threshold': self.thresh}

    def set_params(self, threshold) -> PeriodCancellationEstimator:
        self.thresh = threshold

        return self

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        self._fit_model = RandomForestClassifier(random_state=0).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = 1 - self._fit_model.predict_proba(X)[:, 0]
        return probs > self.thresh

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        pass

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray):
        RocCurveDisplay.from_estimator(self._fit_model, X, y)

    def score(self, X: pd.DataFrame, y: pd.Series):
        return f1_score(y, self._predict(X), average='macro')
