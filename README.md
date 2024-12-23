# Brain Representations
A repository for topological comparisons of dimension reduction algorithms applied to resting-state fMRI data.

## Overview

This repository houses the code underlying the analysis in [5], preprinted [here](https://arxiv.org/abs/2306.13802) on the arXiv. It implements an adaption (see [interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap)) of [4] on neuroimaging data, leveraging the computational efficiency of [Ripser](https://github.com/Ripser/ripser/tree/image-persistence-simple) [2] and [Ripser-image](https://github.com/Ripser/ripser/tree/tight-representative-cycles) [3]. This repository's code is built to compare different "brain representations" (i.e., reduced-data representations) of resting-state fMRI data in the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) (HCP) produced through different algorithms and feature selection choices (see "Data" section). Brain representations are compared by the "metric" embedding they induce on the set of HCP subjects (given some choice of subject-pairwise dissimilarity measure), and the stability of these induced topologies is measured (in part) by the topological bootstrap [1,4]. The `brain_representations` repository contains the code necessary to repeat the analysis conducted in [5]. 

The subdirectories of this repository are listed below, grouped approximately by their role.

## Source Code
Centrally houses code base for project: calculation, visualization, and key scripting functions are found here. Note that all code in these directories enter the pipeline *after* data reduction and featurization. Each directory and subdirectory contain READMEs with further details.

#### `src_py` 
Python repository: distance and persistent homology calculations, statistical analysis, and visualization

#### `src_bash`
Bash repository: distributed SLURM scripting at problem scale, also contains only direct calls to [Ripser](https://github.com/Ripser/ripser/tree/image-persistence-simple) [2] and [Ripser-image](https://github.com/Ripser/ripser/tree/tight-representative-cycles) [3] 

## Data 
Brain representation computation, extraction, and featurization. Note that no subject data of any kind is included in this public repository! Instead, the following directories and their subdirectories contain the extraction/computation/processing code used to standardize brain representations for persistent homology analysis.
#### `profumo_reps`
code for the computing FC network matrices (correlations between timecourses) and spatial correlation matrices (correlations between maps) from rfMRI data 
#### `gradient_reps`
code for the computing diffusion-network based gradient representations from connectivity data at both the subject and group level and derivative features of these representations
#### `ICA_reps`
code for computing FC network matrices, partial FC matrices, and amplitude features from extracted ICA-DR data
#### `glasser`
code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from extracted Glasser-parcellated rfMRI data
#### `schaefer`
code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from Schaefer-parcellated rfMRI data
#### `yeo`
code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from Yeo-parcellated rfMRI data

## Preparations

### Compiling the C++ programs (required for cycle registration)
Before running the code to perform cycle matching [1] in this repository, one needs to compile the C++ files in the `modified ripser` folder. For that
- Install a C++ compiler in your computer. We recommend getting the compiler [GCC](https://gcc.gnu.org/).
- The relevant Makefiles are included in the corresponding folders, so the compilation can be done by running the command line `make` in a terminal opened in the folder. 
- The compiled files should be in the same directory than the python scripts/notebooks in which the cycle matching [1] code is invoked.

### Python dependencies
This repo leverages several Python packages beyond built-ins. These dependencies are listed below, grouped by approximate function and which subdirectories of `src_py` rely on that dependency.

#### General purpose
- [numpy](https://numpy.org/)

#### Parsing Neuroimaging Data: `src_py/HCP_utils.py`
- [nibabel](https://nipy.org/nibabel/) 
	
#### Computing Metric Matrices: `src_py/calclute`
- [pyRiemann](https://pyriemann.readthedocs.io/en/latest)
- [scikit-palm](https://github.com/jameschapman19/scikit-palm)

#### Persistent Homology
- [interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap) (adapted from [4])

#### Figures and post-hoc Analysis: `src_py/figures`
- [python optimal transport library](https://pythonot.github.io/index.html)
- [scipy](https://scipy.org/)
- [pandas](https://pandas.pydata.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/stable/index.html)
	
#### Regularized Canonical Correlation Analysis: `src_py/lindecomp`
- [rcca](https://github.com/gallantlab/pyrcca)
- [scikit-learn](https://scikit-learn.org/stable/)

## Academic use

This code is available and is fully adaptable for individual user customization. If you clone or fork this repository, please include the following bibtex citation: 

```tex
@misc{easley2023comparingrepresentationshighdimensionaldata,
      title={Comparing representations of high-dimensional data with persistent homology: a case study in neuroimaging}, 
      author={Ty Easley and Kevin Freese and Elizabeth Munch and Janine Bijsterbosch},
      year={2023},
      eprint={2306.13802},
      archivePrefix={arXiv},
      primaryClass={cs.CG},
      url={https://arxiv.org/abs/2306.13802}, 
}
```

If your clone or fork includes the [interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap) submodule, please additionally include the following bibtex citation:

```tex
@misc{garcia-redondo_fast_2022,
	title = {Fast {Topological} {Signal} {Identification} and {Persistent} {Cohomological} {Cycle} {Matching}},
	url = {http://arxiv.org/abs/2209.15446},
	urldate = {2022-10-03},
	publisher = {arXiv},
	author = {García-Redondo, Inés and Monod, Anthea and Song, Anna},
	month = sep,
	year = {2022},
	note = {arXiv:2209.15446 [math, stat]},
	keywords = {Mathematics - Algebraic Topology, Statistics - Machine Learning},
}
```

## References
[1] Reani, Yohai, and Omer Bobrowski. 2021. ‘Cycle Registration in Persistent Homology with Applications in Topological Bootstrap’, January. https://arxiv.org/abs/2101.00698v1.

[2] Bauer, Ulrich. 2021. ‘Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes’. Journal of Applied and Computational Topology 5 (3): 391–423. https://doi.org/10.1007/s41468-021-00071-5.

[3] Bauer, Ulrich, and Maximilian Schmahl. 2022. ‘Efficient Computation of Image Persistence’. ArXiv:2201.04170 [Cs, Math], January. http://arxiv.org/abs/2201.04170.

[4] I. García-Redondo, A. Monod, and A. Song, “Fast Topological Signal Identification and Persistent Cohomological Cycle Matching.” arXiv, Sep. 30, 2022. doi: [10.48550/arXiv.2209.15446](https://arxiv.org/abs/2209.15446).

[5] T. Easley, K. Freese, E. Munch, and J. Bijsterbosch, “Comparing representations of high-dimensional data with persistent homology: a case study in neuroimaging,” Nov. 23, 2023, arXiv: arXiv:2306.13802. doi: [10.48550/arXiv.2306.13802](https://arxiv.org/abs/2306.13802).
