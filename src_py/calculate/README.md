# Computational submodule

Primarily calculates metric (Gram) matrices for persistent homology input and secondary/post-hoc analysis. Submodule descriptions are below, grouped by approximate function.

## Metric (Gram) matrix computation
From lists of paths to neuroimaging (rfMRI)-derived datasets, produces subject-by-subject metric (Gram) matrices pre-processed for submission to the [interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap) pipeline.

#### `check.py` 
#### `comp_sim_mtx.py`
#### `permuteHCP.py` 
#### `pyriemann_addons.py`
#### `subj_data_metrics.py`


## Post-hoc analysis
Computes (variant) Wasserstein distances between null-permuted and/or cycle-matched persistence diagrams. 

#### `comp_bootstrap_dists.py`
#### `comp_permtest_dists.py`  
#### `generate_subindex.py`
