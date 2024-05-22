#!/usr/bin/env python3

"""
We generate data according to setups A & C of Nie & Wager's
"Quasi-Oracle Estimation of Heterogeneous Treatment Effects"
Biometrika 108(2) pp. 299–319 (2021)
"""

from __future__ import annotations

import os

import nie_wager_util as util
import numpy as np
import pandas as pd
import scipy.stats as sp_stats

pd.options.display.width = 100
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 100

whereami = os.path.dirname(os.path.abspath(__file__))

batch_size = 10
n_trials = 100

l_list = ["Causal Forest"] + [
    f"{a}-L. w/ {b}" for a in ["S", "T", "X"] for b in ["LGBM", "Ridge"]
]
f_list = ["initial", "no fitness", "elu", "relu"] + [
    f"trans. fit.-{lam}" for lam in [0, 1, 10, 100]
]
pair_list = [(l, f) for l in l_list for f in f_list]


def generate_records(setup) -> int:
    """generate records for setup `setup` (`A` or `C`)
    returns index of last successfully completed trial
    """
    try:
        records = pd.read_csv(
            os.path.join(whereami, "tmp", f"records_setup_{setup}.csv")
        )
        last_trial = max(records.trial)
        if last_trial + 1 >= n_trials:
            return last_trial
    except FileNotFoundError:
        last_trial = -1

    new_records = list()
    for seed in range(last_trial + 1, last_trial + batch_size + 1):
        print(f"trial {seed}".upper())
        rng = np.random.default_rng(seed=seed)
        data = util.generate_A(rng) if setup == "A" else util.generate_C(rng)
        new_records += util.generate_seed_records(seed, data)
    new_records = pd.DataFrame.from_records(
        new_records,
        columns=[
            "trial",
            "features",
            "causal learner",
            "mean sq error",
            "r2 Y~Φ(X)",
        ],
    )

    try:
        records = pd.concat([records, new_records], axis=0)
    except NameError:
        records = new_records

    records.to_csv(f"tmp/records_setup_{setup}.csv", header=True, index=False)
    return max(records.trial)


if __name__ == "__main__":
    for setup in ["A", "C"]:
        print(f"|setup={setup}|".upper().center(79, "="))

        while generate_records(setup) + 1 < n_trials:
            generate_records(setup)

        results = pd.read_csv(
            os.path.join(whereami, "tmp", f"records_setup_{setup}.csv")
        ).loc[lambda df: df.trial < n_trials]
        pivoted = pd.pivot(
            results,
            index=["causal learner", "features"],
            columns="trial",
            values=["mean sq error"],
        )

        # perform a paired t-test to compare performance on
        # original features vs. performance on transformed features
        pvals = np.nan * np.ones(pivoted.shape[0])
        for idx, tpl in enumerate(pivoted.index):
            learner, features = tpl
            if features != "initial":
                old_feat_results = pivoted.loc[(learner, "initial")]
                new_feat_results = pivoted.loc[(learner, features)]
                pvals[idx] = sp_stats.ttest_rel(
                    old_feat_results,
                    new_feat_results,
                    nan_policy="raise",
                    alternative="two-sided",
                ).pvalue

        pivoted = (
            pivoted.assign(
                avg=pivoted.mean(axis="columns", numeric_only=True).round(3),
                stddev=pivoted.std(axis="columns", numeric_only=True).round(3),
                pvals=pvals,
                pvalue=lambda df: df.pvals.apply(
                    lambda x: np.format_float_scientific(x, precision=1)
                ),
            )[["avg", "stddev", "pvalue"]]
            .loc[pair_list]
            .rename(
                index={
                    **{
                        "elu": "elu (lambda=0)",
                        "relu": "relu (lambda=0), ",
                    },
                    **{
                        f"trans. fit.-{lam}": f"tanh (lambda={lam})"
                        for lam in [0, 1, 10, 100]
                    },
                },
                level=1,
            )
        )

        print(pivoted.to_markdown())
        # print(pivoted.to_latex())

        pivoted_r2 = (
            pd.pivot(
                results.groupby(["trial", "features"])
                .first()
                .drop(columns="causal learner")
                .reset_index(),
                index="features",
                columns="trial",
                values="r2 Y~Φ(X)",
            )
            .assign(
                avg=lambda df: df.mean(axis="columns", numeric_only=True),
                stddev=lambda df: df.std(axis="columns", numeric_only=True),
            )[["avg", "stddev"]]
            .loc[f_list]
            .rename(
                index={
                    **{
                        "elu": "elu (lambda=0)",
                        "relu": "relu (lambda=0), ",
                    },
                    **{
                        f"trans. fit.-{lam}": f"tanh (lambda={lam})"
                        for lam in [0, 1, 10, 100]
                    },
                }
            )
        )

        print(pivoted_r2.round(3).to_markdown())
        # print(pivoted_r2.round(3).to_latex())

"""
===================================|SETUP=A|===================================
|                                        |   ('avg', '') |   ('stddev', '') |   ('pvalue', '') |
|:---------------------------------------|--------------:|-----------------:|-----------------:|
| ('Causal Forest', 'initial')           |         0.194 |            0.146 |        nan       |
| ('Causal Forest', 'no fitness')        |         0.137 |            0.108 |          6e-13   |
| ('Causal Forest', 'elu (lambda=0)')    |         0.134 |            0.105 |          1.9e-13 |
| ('Causal Forest', 'relu (lambda=0), ') |         0.15  |            0.11  |          5.3e-08 |
| ('Causal Forest', 'tanh (lambda=0)')   |         0.141 |            0.106 |          1.4e-11 |
| ('Causal Forest', 'tanh (lambda=1)')   |         0.14  |            0.107 |          9.5e-11 |
| ('Causal Forest', 'tanh (lambda=10)')  |         0.132 |            0.101 |          9.9e-14 |
| ('Causal Forest', 'tanh (lambda=100)') |         0.129 |            0.101 |          2.3e-14 |
| ('S-L. w/ LGBM', 'initial')            |         0.133 |            0.071 |        nan       |
| ('S-L. w/ LGBM', 'no fitness')         |         0.149 |            0.079 |          0.025   |
| ('S-L. w/ LGBM', 'elu (lambda=0)')     |         0.145 |            0.067 |          0.075   |
| ('S-L. w/ LGBM', 'relu (lambda=0), ')  |         0.19  |            0.132 |          8e-06   |
| ('S-L. w/ LGBM', 'tanh (lambda=0)')    |         0.137 |            0.07  |          0.53    |
| ('S-L. w/ LGBM', 'tanh (lambda=1)')    |         0.141 |            0.074 |          0.16    |
| ('S-L. w/ LGBM', 'tanh (lambda=10)')   |         0.145 |            0.085 |          0.13    |
| ('S-L. w/ LGBM', 'tanh (lambda=100)')  |         0.14  |            0.061 |          0.22    |
| ('S-L. w/ Ridge', 'initial')           |         0.099 |            0.073 |        nan       |
| ('S-L. w/ Ridge', 'no fitness')        |         0.089 |            0.07  |          4.6e-05 |
| ('S-L. w/ Ridge', 'elu (lambda=0)')    |         0.087 |            0.06  |          0.00027 |
| ('S-L. w/ Ridge', 'relu (lambda=0), ') |         0.098 |            0.078 |          0.8     |
| ('S-L. w/ Ridge', 'tanh (lambda=0)')   |         0.087 |            0.061 |          0.00015 |
| ('S-L. w/ Ridge', 'tanh (lambda=1)')   |         0.087 |            0.064 |          5.5e-06 |
| ('S-L. w/ Ridge', 'tanh (lambda=10)')  |         0.086 |            0.062 |          9.3e-06 |
| ('S-L. w/ Ridge', 'tanh (lambda=100)') |         0.085 |            0.063 |          1.4e-08 |
| ('T-L. w/ LGBM', 'initial')            |         0.745 |            0.294 |        nan       |
| ('T-L. w/ LGBM', 'no fitness')         |         0.565 |            0.226 |          1.1e-08 |
| ('T-L. w/ LGBM', 'elu (lambda=0)')     |         0.562 |            0.2   |          3.6e-08 |
| ('T-L. w/ LGBM', 'relu (lambda=0), ')  |         0.448 |            0.215 |          7.6e-16 |
| ('T-L. w/ LGBM', 'tanh (lambda=0)')    |         0.549 |            0.197 |          3.7e-09 |
| ('T-L. w/ LGBM', 'tanh (lambda=1)')    |         0.57  |            0.223 |          1.4e-07 |
| ('T-L. w/ LGBM', 'tanh (lambda=10)')   |         0.557 |            0.221 |          5.4e-09 |
| ('T-L. w/ LGBM', 'tanh (lambda=100)')  |         0.539 |            0.209 |          1.3e-09 |
| ('T-L. w/ Ridge', 'initial')           |         0.815 |            0.334 |        nan       |
| ('T-L. w/ Ridge', 'no fitness')        |         0.383 |            0.2   |          1.5e-30 |
| ('T-L. w/ Ridge', 'elu (lambda=0)')    |         0.355 |            0.199 |          1.9e-29 |
| ('T-L. w/ Ridge', 'relu (lambda=0), ') |         0.278 |            0.422 |          4.3e-18 |
| ('T-L. w/ Ridge', 'tanh (lambda=0)')   |         0.354 |            0.194 |          6.5e-31 |
| ('T-L. w/ Ridge', 'tanh (lambda=1)')   |         0.371 |            0.212 |          2e-31   |
| ('T-L. w/ Ridge', 'tanh (lambda=10)')  |         0.36  |            0.208 |          2.6e-34 |
| ('T-L. w/ Ridge', 'tanh (lambda=100)') |         0.341 |            0.199 |          1.7e-33 |
| ('X-L. w/ LGBM', 'initial')            |         0.453 |            0.189 |        nan       |
| ('X-L. w/ LGBM', 'no fitness')         |         0.35  |            0.167 |          1.6e-07 |
| ('X-L. w/ LGBM', 'elu (lambda=0)')     |         0.351 |            0.15  |          2.9e-07 |
| ('X-L. w/ LGBM', 'relu (lambda=0), ')  |         0.317 |            0.175 |          3.4e-09 |
| ('X-L. w/ LGBM', 'tanh (lambda=0)')    |         0.352 |            0.149 |          3.1e-07 |
| ('X-L. w/ LGBM', 'tanh (lambda=1)')    |         0.357 |            0.158 |          1.3e-06 |
| ('X-L. w/ LGBM', 'tanh (lambda=10)')   |         0.355 |            0.19  |          3.3e-06 |
| ('X-L. w/ LGBM', 'tanh (lambda=100)')  |         0.335 |            0.146 |          2.6e-09 |
| ('X-L. w/ Ridge', 'initial')           |         0.692 |            0.287 |        nan       |
| ('X-L. w/ Ridge', 'no fitness')        |         0.336 |            0.188 |          3.3e-29 |
| ('X-L. w/ Ridge', 'elu (lambda=0)')    |         0.308 |            0.181 |          4e-29   |
| ('X-L. w/ Ridge', 'relu (lambda=0), ') |         0.257 |            0.427 |          1.6e-14 |
| ('X-L. w/ Ridge', 'tanh (lambda=0)')   |         0.309 |            0.17  |          3.2e-31 |
| ('X-L. w/ Ridge', 'tanh (lambda=1)')   |         0.324 |            0.192 |          2e-30   |
| ('X-L. w/ Ridge', 'tanh (lambda=10)')  |         0.31  |            0.182 |          3.8e-34 |
| ('X-L. w/ Ridge', 'tanh (lambda=100)') |         0.304 |            0.211 |          4.6e-33 |
| features          |   avg |   stddev |
|:------------------|------:|---------:|
| initial           | 0.936 |    0.059 |
| no fitness        | 0.899 |    0.065 |
| elu (lambda=0)    | 0.893 |    0.067 |
| relu (lambda=0),  | 0.706 |    0.083 |
| tanh (lambda=0)   | 0.893 |    0.071 |
| tanh (lambda=1)   | 0.897 |    0.062 |
| tanh (lambda=10)  | 0.893 |    0.061 |
| tanh (lambda=100) | 0.894 |    0.066 |
===================================|SETUP=C|===================================
|                                        |   ('avg', '') |   ('stddev', '') |   ('pvalue', '') |
|:---------------------------------------|--------------:|-----------------:|-----------------:|
| ('Causal Forest', 'initial')           |         0.038 |            0.04  |        nan       |
| ('Causal Forest', 'no fitness')        |         0.028 |            0.027 |          0.00015 |
| ('Causal Forest', 'elu (lambda=0)')    |         0.028 |            0.026 |          0.00022 |
| ('Causal Forest', 'relu (lambda=0), ') |         0.031 |            0.027 |          0.02    |
| ('Causal Forest', 'tanh (lambda=0)')   |         0.029 |            0.028 |          0.00021 |
| ('Causal Forest', 'tanh (lambda=1)')   |         0.03  |            0.027 |          0.0017  |
| ('Causal Forest', 'tanh (lambda=10)')  |         0.029 |            0.027 |          0.00087 |
| ('Causal Forest', 'tanh (lambda=100)') |         0.031 |            0.024 |          0.0095  |
| ('S-L. w/ LGBM', 'initial')            |         0.225 |            0.068 |        nan       |
| ('S-L. w/ LGBM', 'no fitness')         |         0.206 |            0.058 |          0.009   |
| ('S-L. w/ LGBM', 'elu (lambda=0)')     |         0.206 |            0.062 |          0.0096  |
| ('S-L. w/ LGBM', 'relu (lambda=0), ')  |         0.204 |            0.059 |          0.0033  |
| ('S-L. w/ LGBM', 'tanh (lambda=0)')    |         0.208 |            0.063 |          0.012   |
| ('S-L. w/ LGBM', 'tanh (lambda=1)')    |         0.214 |            0.07  |          0.15    |
| ('S-L. w/ LGBM', 'tanh (lambda=10)')   |         0.206 |            0.059 |          0.013   |
| ('S-L. w/ LGBM', 'tanh (lambda=100)')  |         0.209 |            0.058 |          0.034   |
| ('S-L. w/ Ridge', 'initial')           |         0.018 |            0.023 |        nan       |
| ('S-L. w/ Ridge', 'no fitness')        |         0.018 |            0.024 |          0.075   |
| ('S-L. w/ Ridge', 'elu (lambda=0)')    |         0.018 |            0.023 |          0.32    |
| ('S-L. w/ Ridge', 'relu (lambda=0), ') |         0.02  |            0.025 |          0.0023  |
| ('S-L. w/ Ridge', 'tanh (lambda=0)')   |         0.017 |            0.023 |          0.56    |
| ('S-L. w/ Ridge', 'tanh (lambda=1)')   |         0.018 |            0.024 |          0.058   |
| ('S-L. w/ Ridge', 'tanh (lambda=10)')  |         0.018 |            0.023 |          0.66    |
| ('S-L. w/ Ridge', 'tanh (lambda=100)') |         0.018 |            0.024 |          0.0034  |
| ('T-L. w/ LGBM', 'initial')            |         0.575 |            0.14  |        nan       |
| ('T-L. w/ LGBM', 'no fitness')         |         0.541 |            0.116 |          0.038   |
| ('T-L. w/ LGBM', 'elu (lambda=0)')     |         0.542 |            0.124 |          0.056   |
| ('T-L. w/ LGBM', 'relu (lambda=0), ')  |         0.494 |            0.122 |          1.1e-05 |
| ('T-L. w/ LGBM', 'tanh (lambda=0)')    |         0.551 |            0.12  |          0.17    |
| ('T-L. w/ LGBM', 'tanh (lambda=1)')    |         0.55  |            0.137 |          0.14    |
| ('T-L. w/ LGBM', 'tanh (lambda=10)')   |         0.538 |            0.142 |          0.022   |
| ('T-L. w/ LGBM', 'tanh (lambda=100)')  |         0.55  |            0.138 |          0.15    |
| ('T-L. w/ Ridge', 'initial')           |         0.181 |            0.074 |        nan       |
| ('T-L. w/ Ridge', 'no fitness')        |         0.117 |            0.074 |          5.4e-22 |
| ('T-L. w/ Ridge', 'elu (lambda=0)')    |         0.121 |            0.063 |          2.1e-25 |
| ('T-L. w/ Ridge', 'relu (lambda=0), ') |         0.08  |            0.045 |          2.6e-29 |
| ('T-L. w/ Ridge', 'tanh (lambda=0)')   |         0.133 |            0.076 |          7.2e-18 |
| ('T-L. w/ Ridge', 'tanh (lambda=1)')   |         0.114 |            0.067 |          4e-31   |
| ('T-L. w/ Ridge', 'tanh (lambda=10)')  |         0.116 |            0.062 |          3.9e-25 |
| ('T-L. w/ Ridge', 'tanh (lambda=100)') |         0.113 |            0.065 |          1.9e-29 |
| ('X-L. w/ LGBM', 'initial')            |         0.324 |            0.088 |        nan       |
| ('X-L. w/ LGBM', 'no fitness')         |         0.309 |            0.085 |          0.15    |
| ('X-L. w/ LGBM', 'elu (lambda=0)')     |         0.292 |            0.079 |          0.0023  |
| ('X-L. w/ LGBM', 'relu (lambda=0), ')  |         0.277 |            0.083 |          2.3e-05 |
| ('X-L. w/ LGBM', 'tanh (lambda=0)')    |         0.322 |            0.088 |          0.88    |
| ('X-L. w/ LGBM', 'tanh (lambda=1)')    |         0.311 |            0.093 |          0.27    |
| ('X-L. w/ LGBM', 'tanh (lambda=10)')   |         0.301 |            0.097 |          0.038   |
| ('X-L. w/ LGBM', 'tanh (lambda=100)')  |         0.311 |            0.089 |          0.23    |
| ('X-L. w/ Ridge', 'initial')           |         0.168 |            0.069 |        nan       |
| ('X-L. w/ Ridge', 'no fitness')        |         0.103 |            0.068 |          3.7e-25 |
| ('X-L. w/ Ridge', 'elu (lambda=0)')    |         0.108 |            0.061 |          1.9e-27 |
| ('X-L. w/ Ridge', 'relu (lambda=0), ') |         0.067 |            0.041 |          3e-32   |
| ('X-L. w/ Ridge', 'tanh (lambda=0)')   |         0.119 |            0.072 |          3.4e-19 |
| ('X-L. w/ Ridge', 'tanh (lambda=1)')   |         0.101 |            0.063 |          1.2e-33 |
| ('X-L. w/ Ridge', 'tanh (lambda=10)')  |         0.103 |            0.06  |          1e-27   |
| ('X-L. w/ Ridge', 'tanh (lambda=100)') |         0.099 |            0.063 |          1e-31   |
| features          |   avg |   stddev |
|:------------------|------:|---------:|
| initial           | 0.915 |    0.015 |
| no fitness        | 0.928 |    0.013 |
| elu (lambda=0)    | 0.927 |    0.015 |
| relu (lambda=0),  | 0.908 |    0.016 |
| tanh (lambda=0)   | 0.928 |    0.014 |
| tanh (lambda=1)   | 0.927 |    0.014 |
| tanh (lambda=10)  | 0.928 |    0.013 |
| tanh (lambda=100) | 0.929 |    0.014 |
"""
