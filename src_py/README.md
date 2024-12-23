# Python source repository

Primary source repository. Computes metric (Gram) matrices for persistent homology input, performs cycle-matching, and carries out secondary and post-hoc analysis of results. Each module listed below has its own README with further detais.

- `interval-matching_bootstrap` 
[submodule repository](https://github.com/tyo8/interval-matching_bootstrap) forked from original Song and Garcia-Redondo's [interval-matching](https://github.com/inesgare/interval-matching) repository

- `calculate`
Computes finite metric (Gram) matrices from neuroimaging feature data as Ripser/Ripser-image input. Also houses distance computatiosn between cycle-matched diagrams. 

- `figures`
Visualizes/summarizes post-hoc analysis of persistence outputs (including cycle-matched persistence)

- `lindecomp`
Module for regularized CCA analysis of brain representations; linear (affine) method for comparison to the persistent homology approach.

- `HCP_utils.py`
Python utilities for loading, cleaning, and formatting neuroimaging (rfMRI) data from the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (HCP) 1200 Young Adult Subjects release. Dependency for both `calculate` and `figures` submodules.

- `diagram_distances.py `
define and compute distance metrics (i.e., Wasserstein variants) between persitence diagrams. Dependency for both `calculate` and `figures` submodules.

- `toy_models.py`
submodule for creation of several toy-model validation spaces (concentric circles, S<sup>1</sup> wegde S<sup>2</sup> wedge S<sup>1</sup>, S<sup>2</sup> with a diameter, and the torus T<sup>2</sup>)

- `_deprecated`
Included for completeness and transparency/access to version history. 
