[![DOI](https://zenodo.org/badge/804098847.svg)](https://zenodo.org/doi/10.5281/zenodo.11243921)

### Neuroevolutionary representations for learning heterogeneous treatment effects

> Within the field of causal inference, we consider the problem of estimating
> heterogeneous treatment effects from data. We propose and validate a novel
> approach for learning feature representations to aid the estimation of the
> conditional average treatment effect or CATE. Our method focuses on an
> intermediate layer in a neural network trained to predict the outcome from
> the features. In contrast to previous approaches that encourage the
> distribution of representations to be treatment-invariant, we leverage a
> genetic algorithm that optimizes over representations useful for predicting
> the outcome to select those less useful for predicting the treatment. This
> allows us to retain information within the features useful for predicting the
> outcome even if that information may be related to treatment assignment. We
> validate our method on synthetic examples and illustrate its use on a real
> life dataset.

This code accompanies our paper[^1] and is broken into folders:

- `casual_framework/` defines a causal dataset structure and causal
  metalearners that use standard regression methods to perform inference, along
  with our machinery for feature engineering

- `generated_examples/` reproduces numerical experiments described in the paper
  on synthetic data

The code can be run in a [conda](https://conda.io) environment —

```sh
conda env create -f environment.yml
conda activate pearl
python3 experiments_synthetic/nie_wager.py
```

[^1]:
    M. Burkhart and G. Ruiz, _Neuroevolutionary representations for learning
    heterogeneous treatment effects,_ Journal of Computational Science 71
    (2023)

<!---

To update the [environment.yml](./environment.yml) file —

```
conda env export | grep -v "prefix" | grep . > environment.yml
echo "variables:" >> environment.yml
echo "  TF_CPP_MIN_LOG_LEVEL: 3" >> environment.yml
echo "  PYTHONHASHSEED: 0" >> environment.yml
```

-->
