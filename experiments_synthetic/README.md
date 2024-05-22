### Numerical experiments with synthetic data

We explore examples from Nie & Wager [^1], along with some custom examples we
cooked up ourselves. Note that setup B modeled a controlled randomized trial
and setup D had unrelated treatment and control arms. In both cases, the
original features are already independent from the treatment assignment $W$, so
that the two aims of our representation are already satisfied. Thus, we focus
on setups A & C.

_To reproduce this experiment, clear the contents of [tmp/](./tmp/) before
running `nie_wager.py`_ Expect some variability in the results due to random
seeding and parallelism.

[^1]:
    Nie X. & Wager S.,
    [Quasi-oracle estimation of heterogeneous treatment effects](https://doi.org/10.1093/biomet/asaa076),
    Biometrika 108 (2021)
