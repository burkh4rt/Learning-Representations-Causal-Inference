"""
We define a causal meta-learning algorithm that takes a base learner,
and then instantiate some common such algorithms
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class MetaLearner(BaseEstimator, metaclass=ABCMeta):
    """
    a metalearner takes a base learner and provides methods
    to fit(X,W,Y) on training (features, treatments, outcomes)
    where rows correspond to observations
    and a method to predict(Xnew) on new features
    """

    def __init__(self, base_learner) -> None:
        self.base_learner = base_learner

    @abstractmethod
    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        pass


class SLearner(MetaLearner):
    """
    'Single' learner that models treatment as a variable
    """

    def __str__(self) -> str:
        return f"S-L. w/ {self.base_learner()}"

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        self.modelS = self.base_learner().fit(
            np.column_stack((X, W.astype("float"))), Y
        )

    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        pred0 = self.modelS.predict(
            np.column_stack((Xnew, np.zeros((Xnew.shape[0], 1))))
        ).ravel()
        pred1 = self.modelS.predict(
            np.column_stack((Xnew, np.ones((Xnew.shape[0], 1))))
        ).ravel()
        return pred1 - pred0


class TLearner(MetaLearner):
    """
    'Two' learner that models outcomes for the two treatments separately
    """

    def __str__(self) -> str:
        return f"T-L. w/ {self.base_learner()}"

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        self.model0 = self.base_learner().fit(
            X[W.ravel() == 0, :],
            Y[W.ravel() == 0, :],
        )
        self.model1 = self.base_learner().fit(
            X[W.ravel() == 1, :],
            Y[W.ravel() == 1, :],
        )

    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        pred0 = self.model0.predict(Xnew).ravel()
        pred1 = self.model1.predict(Xnew).ravel()
        return pred1 - pred0


class XLearner(MetaLearner):
    """
    X-learner as described in Künzel, Sekhon, Bickel, & Yu's "Metalearners for
     estimating heterogeneous treatment effects using machine learning."
    Proc. Natl. Acad. Sci. U.S.A. 116(10) pp. 4156-4165 (2019)
    """

    def __str__(self) -> str:
        return f"X-L. w/ {self.base_learner()}"

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        model0 = self.base_learner().fit(
            X[W.ravel() == 0, :],
            Y[W.ravel() == 0, :],
        )
        model1 = self.base_learner().fit(
            X[W.ravel() == 1, :],
            Y[W.ravel() == 1, :],
        )
        pred10 = model0.predict(X[W.ravel() == 1, :]).ravel()
        pred01 = model1.predict(X[W.ravel() == 0, :]).ravel()
        pte0 = pred01 - Y[W.ravel() == 0, :].ravel()
        pte1 = Y[W.ravel() == 1, :].ravel() - pred10
        self.model0 = self.base_learner().fit(
            X[W.ravel() == 0, :],
            pte0,
        )
        self.model1 = self.base_learner().fit(
            X[W.ravel() == 1, :],
            pte1,
        )

    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        tau0 = self.model0.predict(Xnew).ravel()
        tau1 = self.model1.predict(Xnew).ravel()
        return (tau0 + tau1) / 2


class CausalForest(MetaLearner):
    """
    Causal Forest Learner from R's grf package
    """

    def __init__(self, base_learner=None) -> None:
        super().__init__(base_learner)

    def __str__(self) -> str:
        return "Causal Forest"

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray):
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        grf = importr("grf")
        self.cf = grf.causal_forest(X=X, Y=Y, W=W, seed=0)

    def predict(self, Xnew: np.ndarray) -> np.ndarray:
        import rpy2.robjects as robjects

        preds = np.asfarray(
            np.array(robjects.r.predict(self.cf, Xnew)).reshape(-1, 1)
        )
        return preds
