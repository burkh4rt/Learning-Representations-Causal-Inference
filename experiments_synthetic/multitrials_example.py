#!/usr/bin/env python3

"""
Looks into finding a map Φ for features X
given a causal dataset.
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from causal_framework import causal_dataset as causal_d
from causal_framework import causal_meta_learners as m_learners
from causal_framework import linear_evolutionary_feature_engineer as featurizer
from causal_framework import regressor_learners as learners

tf.get_logger().setLevel("ERROR")


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
    model = m_learners.CausalForest(base_learner=learners.BaseLearner)
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


def generate_data() -> causal_d.CausalDataset:
    """
    generates a causal dataset
    :return: a causal dataset formed according to the recipe described
    """
    X = np.random.normal(size=[200, 7])
    sigmoid = lambda x: np.divide(np.exp(x), np.exp(x) + 1.0)
    W = np.random.binomial(
        1, sigmoid(7 - np.sum(np.square(X[:, :2]), axis=1))
    )[:, None]
    Y = (
        np.sum(X[:, 2:], axis=1)[:, None]
        + 1 * W
        + np.random.normal(size=W.shape)
    )
    cate = 1 * np.ones_like(Y)

    data = causal_d.CausalDatasetFromArrays(
        features=X, treatment=W, outcome=Y, true_treatment_effect=cate
    )

    return data


def main():
    records = list()
    for seed in range(10):
        print(f"trial {seed}".upper())
        tf.random.set_seed(seed)
        np.random.seed(seed)

        # generate new random dataset
        data = generate_data()

        # perform modelling with initial features
        records_init = train_and_evaluate_meta_learners(data)
        for record in records_init:
            records.append((seed, "initial", *record))

        # perform modelling with transformed features
        phi = featurizer.EvolutionaryFeatureEngineer(
            causal_dataset=data,
            latent_dim=5,
            cohort_size=10,
            num_progenitors=2,
            num_cohorts=5,
            dropout_rate=0.1,
            verbosity=1,
        ).engineer_feature_map()
        modified_dataset = copy.deepcopy(data)
        modified_dataset.features = phi(data.features)
        records_transformed = train_and_evaluate_meta_learners(
            modified_dataset
        )
        for record in records_transformed:
            records.append((seed, "transformed", *record))

        # perform modelling with transformed features
        # *not* using the fitness function
        phi_no_fitness = featurizer.EvolutionaryFeatureEngineer(
            causal_dataset=data,
            latent_dim=5,
            cohort_size=10,
            num_progenitors=2,
            num_cohorts=2,
            dropout_rate=0.1,
            fitness=False,
        ).engineer_feature_map()
        modified_dataset2 = copy.deepcopy(data)
        modified_dataset2.features = phi_no_fitness(data.features)
        records_transformed = train_and_evaluate_meta_learners(
            modified_dataset2
        )
        for record in records_transformed:
            records.append((seed, "no fitness", *record))

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
TRIAL 0
max fitness of cohort 0: 0.951
max fitness of cohort 1: 0.951
max fitness of cohort 2: 0.970
max fitness of cohort 3: 0.970
max fitness of cohort 4: 0.970
TRIAL 1
max fitness of cohort 0: 0.901
max fitness of cohort 1: 0.901
max fitness of cohort 2: 0.994
max fitness of cohort 3: 0.994
max fitness of cohort 4: 0.994
TRIAL 2
max fitness of cohort 0: 0.931
max fitness of cohort 1: 0.931
max fitness of cohort 2: 0.931
max fitness of cohort 3: 1.054
max fitness of cohort 4: 1.055
TRIAL 3
max fitness of cohort 0: 0.875
max fitness of cohort 1: 1.060
max fitness of cohort 2: 1.060
max fitness of cohort 3: 1.060
max fitness of cohort 4: 1.260
TRIAL 4
max fitness of cohort 0: 1.132
max fitness of cohort 1: 1.186
max fitness of cohort 2: 1.186
max fitness of cohort 3: 1.186
max fitness of cohort 4: 1.186
TRIAL 5
max fitness of cohort 0: 1.036
max fitness of cohort 1: 1.179
max fitness of cohort 2: 1.180
max fitness of cohort 3: 1.180
max fitness of cohort 4: 1.179
TRIAL 6
max fitness of cohort 0: 0.816
max fitness of cohort 1: 0.946
max fitness of cohort 2: 0.946
max fitness of cohort 3: 0.946
max fitness of cohort 4: 1.083
TRIAL 7
max fitness of cohort 0: 0.873
max fitness of cohort 1: 0.873
max fitness of cohort 2: 0.874
max fitness of cohort 3: 0.873
max fitness of cohort 4: 0.914
TRIAL 8
max fitness of cohort 0: 0.912
max fitness of cohort 1: 0.912
max fitness of cohort 2: 0.912
max fitness of cohort 3: 0.912
max fitness of cohort 4: 0.912
TRIAL 9
max fitness of cohort 0: 0.901
max fitness of cohort 1: 0.902
max fitness of cohort 2: 0.901
max fitness of cohort 3: 0.909
max fitness of cohort 4: 0.983
...
trial                         avg     sd
causal learner features
Causal Forest  initial      1.058  1.358
               no fitness   0.485  0.540
               transformed  0.465  0.591
"""
