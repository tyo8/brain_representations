# Visualization submodule

Visualizes and summarizes secondary and post-hoc analyses of persistence data.

### `bootstrap_dists.py`
Visually summarizes Wasserstein and/or Wasserstein-variant distances between persistence diagrams and their null-permuted counterparts. Also plots persistence/prevalence of real vs. null-permuted data as function of embedding dimension and noise level when given output produced by `toy_models.py` for validation.

### `compare_topostats.py`  
Visually summarizes Wasserstein and/or Wasserstein-variant distances between many pairs of persistence diagrams composed of cycle-matched generators. Also plots prevalence vs. persistence.

### `distributional_summaries.py`
Visually summarizes distributions of topological features/derived quantities using line plots and categorical swarm plots.

### `prevwt_PD.py`
Creates prevalence-weighted persistence diagrams.
