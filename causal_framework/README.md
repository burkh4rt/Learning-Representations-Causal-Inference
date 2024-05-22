### Causal Framework

This folder contains code that describes a causal dataset, along with methods
to perform causal modelling (infer the treatment effect on an individual
level), and engineer features to improve causal modelling.

- `causal_dataset.py` defines a causal dataset as tuples of (features,
  treatment, outcome) and provides functionality to automatically partition the
  datapoints into training, validation, and testing sets. Here,

  - features (X) are real-valued vectors
  - treatment (W) is boolean-valued, corresponding to either control or
    challenger (experimental)
  - outcome (Y) is a scalar

- `regressor_learners.py` are basically just supervised training methods to
  predict Y|X from (X,Y) tuples

- `causal_meta_learners.py` describes ways to leverage `regressor_learners` to
  infer treatment effect for causal datasets

- `linear_evolutationary_feature_engineer.py` contains our experimental module
  that automatically infers a mapping Φ that can be applied to the features X
  so that causal models trained on data (Φ(X),W,Y) will perform better than
  causal models trained on the original data (X,W,Y)

- `aggregate_diagnostic_plots.py` generates plots of average predicted and
  realized treatment effect vs. predicted treatment effect quantile
