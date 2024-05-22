"""
We define a base learner for multivariate regression
(supervised learning of a real-valued output
from vector-valued features)
and instantiate a few examples.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import lightgbm as lgbm
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV as skl_RidgeCV
from sklearn.neural_network import MLPRegressor as skl_MLPRegressor
from sklearn.pipeline import Pipeline as skl_Pipeline
from sklearn.preprocessing import StandardScaler as skl_Scaler

# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)


class BaseLearner(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """
    a base learner constitutes any sort of supervised regression algorithm
    that provides functions fit(X,y) and predict(Xnew)
    """

    @abstractmethod
    def __str__(self) -> str:
        return "BaseLearner"

    @abstractmethod
    def __init__(self) -> None:
        self.regressor = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.regressor.fit(X=X, y=y.ravel())
        return self

    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        return self.regressor.predict(Xnew)


class LgbmRegressor(BaseLearner):
    """
    standard lightGBM regressor with default parameters
    """

    def __str__(self) -> str:
        return "LGBM"

    def __init__(self) -> None:
        self.regressor = lgbm.LGBMRegressor()


class XgbRegressor(BaseLearner):
    """
    standard xgBoost regressor with default parameters
    """

    def __str__(self) -> str:
        return "XGB"

    def __init__(self) -> None:
        self.regressor = xgb.XGBRegressor()


class KerasRegressor(BaseLearner):
    """
    simple keras sequential model
    """

    def __str__(self) -> str:
        return "KerasNN"

    def __init__(self) -> None:
        self.regressor = tf.keras.models.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(
                    15,
                    activation="tanh",
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.GaussianNoise(0.1),
                tf.keras.layers.Dense(
                    1,
                    activation=None,
                    kernel_initializer="glorot_normal",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
            ]
        )

        self.regressor.compile(optimizer="adam", loss="mse")

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.regressor.fit(
            X, y.ravel(), verbose=0, epochs=100, batch_size=2**6
        )
        return self


class RidgeCVRegressor(BaseLearner):
    """
    linear model with cross-validated ridge regularization
    """

    def __str__(self) -> str:
        return "Ridge"

    def __init__(self) -> None:
        self.regressor = skl_Pipeline(
            [("zscore", skl_Scaler()), ("ridge_cv", skl_RidgeCV())]
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.regressor.fit(X, y.ravel())
        return self


class NNRegressorLBFGS(BaseLearner):
    """
    feedforward neural network with LBFGS solver
    """

    def __str__(self) -> str:
        return "LBFGS-NN"

    def __init__(self) -> None:
        self.regressor = skl_MLPRegressor(
            hidden_layer_sizes=(15),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            random_state=42,
            max_iter=10000,
        )

    #  @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.regressor.fit(X, y.ravel())
        return self
