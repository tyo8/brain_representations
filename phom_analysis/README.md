# Persistent homology experiments

Primary experiment directory. Scaffolds persistent homology-based (and related follow-up) analysis of brain representation data.

- `full-scale-expmt`
Script and compute persistence analysis (primarily cycle-matched persistence) for featurized brain representation data (under various choices of dissimilarity measure)

- `null_testing`
Script and compute persistence analysis for "feature"-type and "subject"-type null-permuted versions of featurized brain brain representation data to compare to correspondings entries in `full-scale-expmt`

- `stability_distances`
Script and distribute computations of distance metrics (i.e., Wasserstein variants) between persitence diagrams. Dependency for both `calculate` and `figures` submodules.

- `toy_models`
submodule for creation of several toy-model validation spaces (concentric circles, S<sup>1</sup> wegde S<sup>2</sup> wedge S<sup>1</sup>, S<sup>2</sup> with a diameter, and the torus T<sup>2</sup>)

- `_deprecated`
Included for completeness and transparency/access to version history. 
