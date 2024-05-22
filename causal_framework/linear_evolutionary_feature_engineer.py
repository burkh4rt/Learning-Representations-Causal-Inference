"""
Given a causal dataset of features X, treatments W, & outcomes Y,
attempts to automatically find a function Φ such that
1. Φ(X) is about as good as X for predicting Y
2. Φ(X) is less good than X for predicting W
In effect, Φ attempts to de-correlate X from W.
The hope is that training a causal meta-learner on Φ(X), W, & Y
will lead to a more robust and more accurate predictor
than training on X, W, & Y (for small amounts of data).
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import os
import sys

import numpy as np
import tensorflow as tf

import causal_framework.causal_dataset as causal_d

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)


@contextlib.contextmanager
def no_print():
    """
    defines a context that suppresses python's print function;
    thanks, @djsmith42
    """
    with open(os.devnull, "w") as null:
        saved_stdout = sys.stdout
        sys.stdout = null
        try:
            yield
        finally:
            sys.stdout = saved_stdout


def mum(f):
    """`Mum's the word`
    runs the function f and suppresses printing
    Parameters
    ----------
    f
        function / callable
    Returns
    -------
    f, but quieter
    See Also
    --------
    no_print
        a context manager version of this function
    """

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        with no_print():
            return f(*args, **kwds)

    return wrapper


class EvolutionaryFeatureEngineer(object):
    def __init__(
        self,
        causal_dataset: causal_d.CausalDataset,
        *,
        latent_dim: int = 10,
        cohort_size: int = 4,
        num_progenitors: int = 2,
        num_cohorts: int = 2,
        dropout_rate: float = 0,
        verbosity: int = 0,
        fitness: bool = True,
        fitness_param: float = 0.0,
        inner_activation: str = "tanh",
    ):
        self.data = causal_dataset
        self.latent_dim = latent_dim
        self.cohort_size = cohort_size
        self.num_progenitors = num_progenitors
        self.num_cohorts = num_cohorts
        self.dropout_rate = max(min(dropout_rate, 1.0), 0.0)
        self.verbosity = int(verbosity)
        self.fitness = bool(fitness)
        self.fitness_param = float(fitness_param)
        self.n_training = self.data.training_outcomes.size
        self.learning_rate = 0.1 if self.n_training < 1000 else 0.01
        self.batch_size = 2**6 if self.n_training < 1000 else 2**8
        self.patience = 5 if self.n_training < 1000 else 20
        self.inner_activation = inner_activation

    def _generate_model(self) -> tf.keras.Model:
        """
        trains a model for outcome given features with an intermediate layer
            Φ(features)
        :return: model as described
        """

        input_x = tf.keras.Input(shape=(self.data.n_feat,))
        phi_x = tf.keras.layers.Dense(
            self.latent_dim,
            name="phi_x",
            activation=self.inner_activation,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.5),
        )(input_x)
        phi_x_dropout = tf.keras.layers.AlphaDropout(
            rate=self.dropout_rate, seed=0
        )(phi_x)
        h_phi_x = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
        )(phi_x_dropout)

        model_x_y = tf.keras.Model(inputs=input_x, outputs=h_phi_x)
        model_x_y.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        model_x_y.fit(
            x=self.data.training_features,
            y=self.data.training_outcomes,
            epochs=100,
            batch_size=self.batch_size,
            validation_data=(
                self.data.validation_features,
                self.data.validation_outcomes,
            ),
            verbose=0 if self.verbosity < 2 else 2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.patience, restore_best_weights=True
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        return model_x_y

    def _evaluate_model_fitness(self, model: tf.keras.Model) -> float:
        """
        assigns fitness score to model based on how readily
        one may use Φ(features) to predict recipe
        :param model: supplied model instance
        :return: score > 0
        """
        if not self.fitness:
            # don't use fitness function
            return 0.0

        model.trainable = False

        phi_x_dropout = tf.keras.layers.AlphaDropout(
            rate=self.dropout_rate, seed=0
        )(model.get_layer("phi_x").output)
        hidden_3 = tf.keras.layers.Dense(
            10,
            activation="tanh",
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.5),
        )(phi_x_dropout)
        f_phi_x = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
        )(hidden_3)

        model_phi_w = tf.keras.Model(inputs=model.input, outputs=f_phi_x)
        model_phi_w.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
        )
        model_phi_w.fit(
            x=self.data.training_features,
            y=self.data.training_treatments,
            epochs=100,
            batch_size=self.batch_size,
            validation_data=(
                self.data.validation_features,
                self.data.validation_treatments,
            ),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.patience, restore_best_weights=True
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )

        phi_fitness = model_phi_w.evaluate(
            self.data.validation_features,
            self.data.validation_treatments,
            verbose=0,
        )

        if self.fitness_param > np.finfo(np.float16).eps:
            # add weighted penalty for candidates bad at predicting outcome
            phi_fitness -= self.fitness_param * model.evaluate(
                self.data.validation_features,
                self.data.validation_outcomes,
                verbose=0,
            )

        model.trainable = True
        return phi_fitness

    def _conceive_filial_model(
        self, m1: tf.keras.Model, m2: tf.keras.Model
    ) -> tf.keras.Model:
        """
        randomly combine components of Φ from m1 and m2 --
        cf. Montana & Davis's "Training feedforward neural networks using
        genetic algorithms."
        Proc. Int. Jt. Conf. Artif. Intell. pp. 762–767 (1989)
        :param m1: first model
        :param m2: second model
        :return: the happy offspring of m1 & m2
        """

        # create model architecture as per usual
        input_x = tf.keras.Input(shape=(self.data.n_feat,))
        phi_x = tf.keras.layers.Dense(
            self.latent_dim,
            name="phi_x",
            activation=self.inner_activation,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.5),
        )(input_x)
        phi_x_dropout = tf.keras.layers.AlphaDropout(
            rate=self.dropout_rate, seed=0
        )(phi_x)
        h_phi_x = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer="glorot_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
        )(phi_x_dropout)

        m3 = tf.keras.Model(inputs=input_x, outputs=h_phi_x)
        m3.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        w1, b1 = m1.get_layer("phi_x").get_weights()
        w2, b2 = m2.get_layer("phi_x").get_weights()
        w3, b3 = w1.copy(), b1.copy()
        switch_idx = np.random.binomial(1, 0.5, size=b3.shape).astype("bool")
        w3[:, switch_idx], b3[switch_idx] = w2[:, switch_idx], b2[switch_idx]
        m3.get_layer("phi_x").set_weights([w3, b3])

        m3.training = True
        m3.fit(
            x=self.data.training_features,
            y=self.data.training_outcomes,
            epochs=100,
            batch_size=self.batch_size,
            validation_data=(
                self.data.training_features,
                self.data.training_outcomes,
            ),
            verbose=0 if self.verbosity < 2 else 2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=self.patience, restore_best_weights=True
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        return m3

    @mum
    def _generate_first_cohort(self) -> dict[tf.keras.Model, float]:
        """
        initializes a cohort of models
        :return: dictionary of model:fitness correspondences
        """
        models = [self._generate_model() for _ in range(self.cohort_size)]
        fitnesses = [self._evaluate_model_fitness(m) for m in models]
        return dict(zip(models, fitnesses))

    @mum
    def _generate_new_cohort(
        self, old_cohort: dict[tf.keras.Model, float]
    ) -> dict[tf.keras.Model, float]:
        """
        initialize new cohort with best-performing member of old cohort;
        pair off the best n_progenitors from the old cohort to add to the new
            cohort;
        fill the remaining len(old_cohort) - (n_progenitors choose 2) - 1 spots
            with random offspring
        :param data: causal dataset
        :param old_cohort: list of models belonging to previous generation
        :param n_progenitors: number of top performers to select to populate
            new generation
        :return: new generation as list of models
        """

        progenitors = sorted(
            old_cohort.keys(), key=lambda k: old_cohort[k], reverse=True
        )[: self.num_progenitors]

        new_cohort = [progenitors[0]]
        for m1, m2 in itertools.combinations(progenitors, 2):
            new_cohort.append(self._conceive_filial_model(m1, m2))

        while len(new_cohort) < self.cohort_size:
            new_cohort.append(self._generate_model())

        new_models = new_cohort[: self.cohort_size]
        new_fitnesses = [self._evaluate_model_fitness(m) for m in new_models]
        return dict(zip(new_models, new_fitnesses))

    def engineer_feature_map(self):
        cohort = self._generate_first_cohort()
        if self.verbosity > 0:
            print(
                f"max fitness of cohort 0: {np.max(list(cohort.values())):.3f}"
            )
        for i in range(self.num_cohorts - 1):
            cohort = self._generate_new_cohort(old_cohort=cohort)
            if self.verbosity > 0:
                print(
                    f"max fitness of cohort {i+1}: "
                    f"{np.max(list(cohort.values())):.3f}"
                )

        best_model = max(cohort.keys(), key=lambda m: cohort[m])
        best_model.trainable = False
        best_phi = tf.keras.Model(
            inputs=best_model.input,
            outputs=best_model.get_layer("phi_x").output,
        )
        best_phi.trainable = False

        Phi = lambda features: best_phi.predict(features)
        return Phi
