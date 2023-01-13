# Brain Representations
A (partial) repository for my thesis project

## Outline 
This repository contains code for/commemorates the directory tree structure of the repository underlying the analyses in <at least one untitled future paper.>

In this repository:
1. we have a bunch of bash/slurm code meant to script a bunch of resource-heavy operations on a compute cluster, plus the occasional bit of python code meant to do the same
2. we have a bunch of empty directories referencing data that is too big/too private to fit on github
3. we have a few nuggets of source code nestled away in the mess; the most notable one is 'src_py', with 'src_bash' a distant second and 'src_MATLAB' an even further third
4. We also have Ripser [2] and Ripser-image [3] here, upon which all of the persistence analysis is *actually* built

## Preparations

### Compiling the C++ programs (required for cycle registration)
Before running the code to perform cycle matching [1] in this repository, one needs to compile the C++ files in the `modified ripser` folder. For that
- Install a C++ compiler in your computer. We recommend getting the compiler [GCC](https://gcc.gnu.org/).
- The relevant Makefiles are included in the corresponding folders, so the compilation can be done by running the command line `make` in a terminal opened in the folder. 
- The compiled files should be in the same directory than the python scripts/notebooks in which the cycle matching [1] code is invoked.

### Installing Python libraries
The analyses in this have several Python package dependencies; some are listed below according to their role in the code.

Parsing Neuroimaging Data
- [nibabel](https://nipy.org/nibabel/)
- [nilearn](https://nilearn.github.io/stable/index.html)
	
Canonical Correlation Analysis (regularized)
- [rcca](https://github.com/gallantlab/pyrcca)
	
Persistent Homology
- [interval-matching](https://github.com/tyo8/interval-matching) [4]
- [giotto-tda](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) [5]
	
Background
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/stable/index.html)

## Structure of the repository

This repository is organised as follows.

Source Code
- 'src_py': python code (this does most of the repo's work)
- 'src_bash': bash scripts
- 'src_MATLAB': mostly here to borrow functions from [PALM](https://github.com/andersonwinkler/PALM), which we need to do permutation testing in the CCA analysis on Human Connectome Project data

Experiments
- 'datalists': lists of data files corresponding that help set conditions for specific experiments
- 'metric_tests': **preliminary** testing of the impact of metric choice on persistence data
- 'bootstrap_benchmarks': benchmarking the cycle registration bootstrapping scheme to test scalability
- 'phom_analysis': all persistent homology-based analyses in this project (with the exception of 'metric_tests' and 'bootstrap_benchmarks')
- 'nonPH_analysis': most notably contains output of CCA analysis of shared information between pairs of brain representations

Brain Representation Data/Extraction
- a bunch of other stuff (for adding later)

## Academic use

This code is available and is fully adaptable for individual user customization. If you use the our methods, please cite as the following:

```tex
@misc{ty-o_untitled,
	title = {Unknown},
	publisher = {publisher?},
	author = {Easley, Ty, Munch, Elizabeth, Freese, Kelsey, and Bijsterbosch, Janine},
	month = ??,
	year = {2023/4},
	note = {arXiv:####.<subj> [tag1, tag2]},
	keywords = {topological data analysis, neuroscience, computational topology, persistent homology, functional connectivity},
}
```

## References
[1] Reani, Yohai, and Omer Bobrowski. 2021. ‘Cycle Registration in Persistent Homology with Applications in Topological Bootstrap’, January. https://arxiv.org/abs/2101.00698v1.

[2] Bauer, Ulrich. 2021. ‘Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes’. Journal of Applied and Computational Topology 5 (3): 391–423. https://doi.org/10.1007/s41468-021-00071-5.

[3] Bauer, Ulrich, and Maximilian Schmahl. 2022. ‘Efficient Computation of Image Persistence’. ArXiv:2201.04170 [Cs, Math], January. http://arxiv.org/abs/2201.04170.

[4] I. García-Redondo, A. Monod, and A. Song, “Fast Topological Signal Identification and Persistent Cohomological Cycle Matching.” arXiv, Sep. 30, 2022. doi: 10.48550/arXiv.2209.15446. https://arxiv.org/abs/2209.15446

[5] G. Tauzin et al., “giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration,” in NeurIPS 2020 Workshop TDA and Beyond, 2020. Accessed: Mar. 21, 2021. [Online]. Available: https://github.com/giotto-ai/pyflagser.
