from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, accuracy_score

from IMLearn.base import BaseEstimator


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, threshold: float = None) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()
        self.__fit_model: RandomForestClassifier = None
        self.thresh = threshold

    def get_params(self, deep=False):
        return {'threshold': self.thresh}

    def set_params(self, threshold) -> AgodaCancellationEstimator:
        self.thresh = threshold

        return self

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.__fit_model = RandomForestClassifier(random_state=0).fit(X, y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        probs = self.__fit_model.predict_proba(X)[:, 1]
        return np.logical_not(probs > self.thresh if self.thresh is not None else probs).astype('int')

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        pass

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray):
        RocCurveDisplay.from_estimator(self.__fit_model, X, y)

    def score(self, X: pd.DataFrame, y: pd.Series):
        return accuracy_score(y, self._predict(X))
