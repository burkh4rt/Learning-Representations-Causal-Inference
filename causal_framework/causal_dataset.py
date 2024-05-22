"""
A causal dataset contains rows of observations
with columns including features, a single (boolean) treatment,
and outcome.
We offer two ways to instantiate such a dataset --
(1) load data from a csv file and process in pandas
(2) provide arrays for (features, treatments, outcomes)
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


class CausalDataset(object, metaclass=ABCMeta):
    """
    stores training/validation/testing partition
    of dataset with (multiple) features, treatment, & outcome
    """

    @property
    @abstractmethod
    def n_feat(self):
        pass

    @property
    @abstractmethod
    def training_features(self):
        pass

    @property
    @abstractmethod
    def training_treatments(self):
        pass

    @property
    @abstractmethod
    def training_outcomes(self):
        pass

    @property
    @abstractmethod
    def validation_features(self):
        pass

    @property
    @abstractmethod
    def validation_treatments(self):
        pass

    @property
    @abstractmethod
    def validation_outcomes(self):
        pass

    @property
    @abstractmethod
    def testing_features(self):
        pass

    @property
    @abstractmethod
    def testing_treatments(self):
        pass

    @property
    @abstractmethod
    def testing_outcomes(self):
        pass


class CausalDatasetFromArrays(CausalDataset):
    def __init__(
        self,
        features: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        true_treatment_effect: np.ndarray = None,
        *,
        splitting_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = None,
    ):
        self.data_size = outcome.size
        self.n_features = features.shape[1]
        self.features = features
        self.treatment = treatment
        self.outcome = outcome
        self.true_treatment_effect = true_treatment_effect

        if random_seed is not None:
            np.random.seed(random_seed)

        if len(splitting_ratio) != 3:
            print(f"Please check splitting ratio {splitting_ratio}")
        self.splitting_ratio = np.array(splitting_ratio)
        self.splitting_ratio /= np.sum(self.splitting_ratio)
        self.splitting_cdf = np.cumsum(self.splitting_ratio)

        unif_rnd = np.random.uniform(size=self.data_size)
        self.idx_assignment = np.digitize(unif_rnd, self.splitting_cdf)
        if np.unique(self.idx_assignment).size < 3:
            print("Some subsets of the partition are empty.")

    @property
    def n_feat(self):
        return self.n_features

    @property
    def training_features(self):
        return self.features[self.idx_assignment == 0,]

    @training_features.setter
    def training_features(self, x):
        self.features[self.idx_assignment == 0, :] = x

    @property
    def training_treatments(self):
        return self.treatment[self.idx_assignment == 0]

    @property
    def training_outcomes(self):
        return self.outcome[self.idx_assignment == 0]

    @property
    def validation_features(self):
        return self.features[self.idx_assignment == 1,]

    @validation_features.setter
    def validation_features(self, x):
        self.features[self.idx_assignment == 1, :] = x

    @property
    def validation_treatments(self):
        return self.treatment[self.idx_assignment == 1]

    @property
    def validation_outcomes(self):
        return self.outcome[self.idx_assignment == 1]

    @property
    def testing_features(self):
        return self.features[self.idx_assignment == 2,]

    @testing_features.setter
    def testing_features(self, x):
        self.features[self.idx_assignment == 2,] = x

    @property
    def testing_treatments(self):
        return self.treatment[self.idx_assignment == 2]

    @property
    def testing_outcomes(self):
        return self.outcome[self.idx_assignment == 2]

    @property
    def training_cates(self):
        if self.true_treatment_effect is not None:
            return self.true_treatment_effect[self.idx_assignment == 0]
        else:
            pass

    @property
    def validation_cates(self):
        if self.true_treatment_effect is not None:
            return self.true_treatment_effect[self.idx_assignment == 1]
        else:
            pass

    @property
    def testing_cates(self):
        if self.true_treatment_effect is not None:
            return self.true_treatment_effect[self.idx_assignment == 2]
        else:
            pass


class CausalDatasetFromFile(CausalDatasetFromArrays):
    """
    load a causal dataset from a csv file
    where header names are used to designate
    features/treatment/outcome
    """

    def __init__(
        self,
        file_name: str,
        outcome_col: list[str],
        recipe_col: list[str],
        feature_col: list[str] = None,
        *,
        splitting_ratio: tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = None,
    ):
        # load data
        df = pd.read_csv(file_name, low_memory=False)

        # convert categorical columns to one-hot
        df = pd.get_dummies(df).astype(np.float)

        # handle columns with missing values
        na_columns = [k for k, v in df.isna().any().to_dict().items() if v]
        for k in na_columns:
            df[k + "_isna"] = df[k].isna().astype(np.float)
            df[k] = df[k].fillna(df[k].median(skipna=True))

        if set(outcome_col).issubset(set(df.columns)):
            Y = np.reshape(df[outcome_col].to_numpy(), (-1, 1))
        else:
            print(f"issue with outcome column list {outcome_col}")

        if set(recipe_col).issubset(set(df.columns)):
            W = np.reshape(df[recipe_col].to_numpy().astype(int), (-1, 1))
        else:
            print(f"issue with recipe column list {recipe_col}")

        if feature_col is not None:
            if set(feature_col).issubset(set(df.columns)):
                X = df[feature_col].to_numpy()
            else:
                print(f"issue with feature column list {feature_col}")
        else:
            inferred_feature_col = [
                k for k in df.columns if k not in outcome_col + recipe_col
            ]
            X = df[inferred_feature_col].to_numpy()

        super().__init__(
            features=X,
            treatment=W,
            outcome=Y,
            splitting_ratio=splitting_ratio,
            random_seed=random_seed,
        )
