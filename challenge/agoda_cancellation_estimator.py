from __future__ import annotations
from typing import NoReturn

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from IMLearn.base import BaseEstimator
import numpy as np


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
        return probs > self.thresh if self.thresh is not None else probs

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
