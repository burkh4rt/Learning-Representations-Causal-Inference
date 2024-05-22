#!/usr/bin/env python3

"""
A very simple example of how feature engineering can aid causal inference.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from causal_framework import causal_dataset as causal_d
from causal_framework import causal_meta_learners as m_learners
from causal_framework import linear_evolutionary_feature_engineer as featurizer
from causal_framework import regressor_learners as learners


def generate_data() -> causal_d.CausalDataset:
    """
    creates a causal dataset according to generative specs
    :return: casual dataset
    """
    X = np.random.uniform(low=-1, high=1, size=[200, 1])
    W = np.random.binomial(1, 1 - np.abs(X))
    Y = W * np.sign(X)
    cate = np.sign(X)

    data = causal_d.CausalDatasetFromArrays(
        features=X, treatment=W, outcome=Y, true_treatment_effect=cate
    )
    return data


def train_and_evaluate_meta_learners(
    data: causal_d.CausalDataset,
) -> list[tuple[str, float]]:
    """
    trains causal meta learners on the training portion of the dataset
    then predicts treatment effects for the testing portion of the data
    and reports results
    :param data: causal dataset used for training and testing
    :return: list of tuples containing (causal inference method, testing error)
    """
    trial_records = list()
    for meta in [m_learners.CausalForest]:
        model = meta(base_learner=learners.RidgeCVRegressor)
        model.fit(
            X=data.training_features,
            W=data.training_treatments,
            Y=data.training_outcomes,
        )
        cate_preds = model.predict(Xnew=data.testing_features)
        trial_records.append(
            (
                str(model),
                float(np.mean(np.square(cate_preds - data.testing_cates))),
            )
        )
    return trial_records


def main():
    records = list()

    for seed in range(10):
        np.random.seed(seed)

        data = generate_data()
        records_init = train_and_evaluate_meta_learners(data)
        for record in records_init:
            records.append((seed, "initial", *record))

        data2 = copy.deepcopy(data)
        phi = featurizer.EvolutionaryFeatureEngineer(
            causal_dataset=data,
            latent_dim=10,
            cohort_size=4,
            num_progenitors=2,
            num_cohorts=3,
            dropout_rate=0.1,
        ).engineer_feature_map()
        data2.features = phi(data2.features)
        records_transformed = train_and_evaluate_meta_learners(data2)
        for record in records_transformed:
            records.append((seed, "transformed", *record))

        data3 = copy.deepcopy(data)
        data3.features = np.sign(data3.features)
        records_oracle = train_and_evaluate_meta_learners(data3)
        for record in records_oracle:
            records.append((seed, "oracle", *record))

    results = pd.DataFrame.from_records(
        records,
        columns=["trial", "features", "causal learner", "mean sq error"],
    )
    pivoted = pd.pivot(
        results,
        index=["causal learner", "features"],
        columns="trial",
        values="mean sq error",
    )
    pivoted = pivoted.assign(
        avg=pivoted.mean(axis="columns", numeric_only=True),
        sd=pivoted.std(axis="columns", numeric_only=True),
    )
    print(pivoted.round(3))


if __name__ == "__main__":
    # execute only if run as a script
    main()


# sample output
"""
trial                           0      1      2  ...     9     avg     sd
causal learner features                          ...
Causal Forest  initial      0.147  0.078  0.148  ... 0.146   0.094  0.047
               oracle       0.060  0.023  0.050  ... 0.009   0.028  0.021
               transformed  0.081  0.034  0.103  ... 0.123   0.077  0.044
"""
