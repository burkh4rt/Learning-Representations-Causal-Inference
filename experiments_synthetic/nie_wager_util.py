#!/usr/bin/env python3

"""Functions used to run trials on Nie & Wager's setups

Note the use of `tf.keras.backend.clear_session()` --
https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
-- in order to limit the accumulation of memory usage over independent trials.
"""

from __future__ import annotations

import copy
import itertools
import os
import sys

import numpy as np
import statsmodels.api as sm
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from causal_framework import causal_dataset as causal_d
from causal_framework import causal_meta_learners as m_learners
from causal_framework import linear_evolutionary_feature_engineer as featuriser
from causal_framework import regressor_learners as learners


def generate_A(rng: np.random.Generator) -> causal_d.CausalDataset:
    """
    generates a causal dataset according to Nie & Wager's setup A
    :return: a causal dataset formed according to the recipe described
    """
    n = 200
    d = 24
    sigma = 1.0
    X = rng.uniform(size=[n, d])
    trim_eta = lambda x, eta: np.maximum(eta, np.minimum(x, 1 - eta))
    e_star_X = trim_eta(np.sin(np.pi * X[:, 0] * X[:, 1]), 0.1)
    b_star_X = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * np.square(X[:, 2] - 0.5)
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    t_star_X = (X[:, 0] + X[:, 1]) / 2
    W = rng.binomial(1, e_star_X)
    epsilon = rng.normal(size=[n])
    Y = (b_star_X + (W - 0.5) * t_star_X + sigma * epsilon)[:, None]
    W = W[:, None]
    cate = t_star_X

    data = causal_d.CausalDatasetFromArrays(
        features=X, treatment=W, outcome=Y, true_treatment_effect=cate
    )
    return data


def generate_C(rng: np.random.Generator) -> causal_d.CausalDataset:
    """
    generates a causal dataset according to Nie & Wager's setup C
    :return: a causal dataset formed according to the recipe described
    """
    n = 500
    d = 12
    sigma = 1.0
    X = rng.uniform(size=[n, d])
    e_star_X = np.divide(1.0, 1.0 + np.exp(X[:, 1] + X[:, 2]))
    b_star_X = 2.0 * np.log(1.0 + np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    t_star_X = np.ones_like(X[:, 0])
    W = rng.binomial(1, e_star_X)
    epsilon = rng.normal(size=[n])
    Y = (b_star_X + (W - 0.5) * t_star_X + sigma * epsilon)[:, None]
    W = W[:, None]
    cate = t_star_X

    data = causal_d.CausalDatasetFromArrays(
        features=X, treatment=W, outcome=Y, true_treatment_effect=cate
    )
    return data


def train_and_evaluate_meta_learners(
    data: causal_d.CausalDataset,
) -> list[tuple]:
    """
    trains causal meta learners on the training portion of the dataset
    then predicts treatment effects for the testing portion of the data
    and reports results
    :param data: causal dataset used for training and testing
    :return: list of tuples containing (causal inference method, testing error)
    """
    trial_records = list()
    for meta, learner in list(
        itertools.product(
            [
                m_learners.SLearner,
                m_learners.TLearner,
                m_learners.XLearner,
            ],
            [learners.RidgeCVRegressor, learners.LgbmRegressor],
        )
    ) + [(m_learners.CausalForest, learners.BaseLearner)]:
        model = meta(base_learner=learner)
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


def generate_map_records(phi, data) -> list[tuple]:
    """
    generates results for a given transformation phi and input dataset data
    """
    transformed_dataset = copy.deepcopy(data)
    transformed_dataset.features = phi(transformed_dataset.features)
    records = train_and_evaluate_meta_learners(transformed_dataset)
    return records


def generate_seed_records(
    seed: int, data: causal_d.CausalDataset
) -> list[tuple]:
    """
    generates records for a given seed and dataset formed from that seed
    """
    records = []

    # perform modelling with initial features
    records_init = train_and_evaluate_meta_learners(data)
    y_feat_r2 = (
        sm.OLS(data.testing_outcomes, data.testing_features).fit().rsquared
    )
    for record in records_init:
        records.append((seed, "initial", *record, y_feat_r2))

    for fparam in [0, 1, 10, 100]:
        # perform modelling with transformed features
        phi = featuriser.EvolutionaryFeatureEngineer(
            causal_dataset=data,
            latent_dim=20,
            cohort_size=4,
            num_progenitors=2,
            num_cohorts=5,
            dropout_rate=0.05,
            fitness_param=fparam,
        ).engineer_feature_map()
        records_transformed = generate_map_records(phi, data)
        y_feat_r2 = (
            sm.OLS(data.testing_outcomes, phi(data.testing_features))
            .fit()
            .rsquared
        )
        for record in records_transformed:
            records.append((seed, f"trans. fit.-{fparam}", *record, y_feat_r2))

    # perform modelling with transformed features
    # *not* using the fitness function
    phi_no_fitness = featuriser.EvolutionaryFeatureEngineer(
        causal_dataset=data,
        latent_dim=20,
        cohort_size=1,
        num_progenitors=1,
        num_cohorts=1,  # no point in multiple cohorts if fitness=False
        dropout_rate=0.05,
        fitness=False,
    ).engineer_feature_map()
    records_transformed = generate_map_records(phi_no_fitness, data)
    y_feat_r2 = (
        sm.OLS(data.testing_outcomes, phi_no_fitness(data.testing_features))
        .fit()
        .rsquared
    )
    for record in records_transformed:
        records.append((seed, "no fitness", *record, y_feat_r2))

    # perform modelling with relu inner activation
    phi_relu = featuriser.EvolutionaryFeatureEngineer(
        causal_dataset=data,
        latent_dim=20,
        cohort_size=4,
        num_progenitors=2,
        num_cohorts=2,
        dropout_rate=0.05,
        inner_activation="relu",
    ).engineer_feature_map()
    records_transformed = generate_map_records(phi_relu, data)
    y_feat_r2 = (
        sm.OLS(data.testing_outcomes, phi_relu(data.testing_features))
        .fit()
        .rsquared
    )
    for record in records_transformed:
        records.append((seed, "relu", *record, y_feat_r2))

    # try elu activation
    phi_elu = featuriser.EvolutionaryFeatureEngineer(
        causal_dataset=data,
        latent_dim=20,
        cohort_size=4,
        num_progenitors=2,
        num_cohorts=2,
        dropout_rate=0.05,
        inner_activation="elu",
    ).engineer_feature_map()
    records_transformed = generate_map_records(phi_elu, data)
    y_feat_r2 = (
        sm.OLS(data.testing_outcomes, phi_elu(data.testing_features))
        .fit()
        .rsquared
    )
    for record in records_transformed:
        records.append((seed, "elu", *record, y_feat_r2))

    # gc
    tf.keras.backend.clear_session()

    return records
