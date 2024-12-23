# Regularized CCA submodule

Computes regularized canonical correlation analysis ([rCCA](https://github.com/gallantlab/pyrcca)) between pairs of brain representation (post-featurization) which is used to define a measurement of "maximal shared affine dimensions" between them. Included for completeness -- unused in current version of paper and will likely be moved to `_deprecated` in a future push.

### `brainrep.py`
Computes regularized CCA between pairs of feature sets listed in input file. Regularization hyperparameter chosen by cross-validation unless specified via principal component truncation. Partial Least Squares (PLS) is included as an alternate method to CCA (corresponding to "unwhitened" CCA).

### `dimstab.py`
Defines and measures "maximal shared affine dimension" given (a) rCCA results and (b) permutation-tested rCCA results between two brain representations.

### `permtest.py`  
Conducts pairwise regularized CCA between permuted feature sets.

### `stability.py`
Assesses the stability of rCCA results with respect to variation in the regularization hyperparameter; shown over distribution of chosen hyperparameters.

### `pull_cancorrs.py`  
Aggregates canonical correlation values from rCCA results.

### `analyze_CCA_permtests.py`
Figure/visualization code for analyzing results of pairwise rCCA results, including permutation, stability, and dimensional stability testing.
