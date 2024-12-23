# Computational submodule

Primarily calculates metric (Gram) matrices for persistent homology input and secondary/post-hoc analysis. Submodule descriptions are below, grouped by approximate function.

## Metric (Gram) matrix computation
From lists of paths to neuroimaging (rfMRI)-derived datasets, produces subject-by-subject metric (Gram) matrices pre-processed for submission to the [interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap) pipeline.

#### `check.py`
Simple utility module to verify validity of input assumptions

#### `comp_sim_mtx.py`
Engine for computation of metric (Gram) matrices from neuroimaging feature data. Requires specification of dissimilarity measure and output types, and checks data quality, validity, and resource use during load-in.

#### `permuteHCP.py`
Returns permutation of input data of specified type, either "subject" or "feature"; the data axis corresponding to the permutation type is permuted while the other is held fixed. Subject-type permutations are computed using [skpalm](https://github.com/jameschapman19/scikit-palm) to maintain exchangeability and saved in blocks to ensure repeatability across brain representations. Feature-type permutations are computed in-place on data, each with a specified seed integer (i.e., [np.random.default_rng(seed)](https://numpy.org/doc/stable/reference/random/generator.html)) to ensure repeatability.

#### `pyriemann_addons.py`
Adds a flexible and data-tailored regularization scheme to the `mean_ale` function of [pyRiemann](https://pyriemann.readthedocs.io/en/latest)

#### `subj_data_metrics.py`
Module defining dissimilarity measures between individual data pairs: defines methods `spd_cos`, `inner`, `Frob_dist`, `ztrans_psim`, `geodesic`

#### [skpalm](https://github.com/jameschapman19/scikit-palm)
Submodule to allow permutations of [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (HCP) data that repsect the family structure and preserve exchangeability (dependency for `permuteHCP.py`)


## Post-hoc analysis
Computes (variant) Wasserstein distances between null-permuted and/or cycle-matched persistence diagrams. 

#### `comp_bootstrap_dists.py`
Computes (variant) Wasserstein distances between persistence modules with cycle-matched generators; makes strong assumptions on naming conventions to find matches.

#### `comp_permtest_dists.py`  
Computes (variant) Wasserstein distances between persistence diagrams and their corresponding diagrams from null-permuted data. 

#### `generate_subindex.py`
Simple utility that converts 64bit-encoded ASCII string to/from binary subindex; dependency for `comp_bootstrap_dists.py`
