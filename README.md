# Brain Representations
A repository for topological comparisons of dimension reduction algorithms applied to resting-state fMRI data.

## Rough Overview
This repository contains code for/commemorates the directory tree structure of the repository underlying the analyses in <at least one untitled future paper.>

In this repository:
1. we have several different repositories of source code; the most populated one is 'src_py', with 'src_bash' a distant second and 'src_MATLAB' an even further third
2. we have lots of bash/slurm code meant to script resource-heavy operations on a compute cluster, plus the occasional bit of python code meant to do the same
3. we have brain representation extraction code and figure-making code in relevant directories: code is only included in "src_\*" if it is intentionally multipurpose
4. we have directories corresponding to the computation, outputs, and feature extraction of the brain representation/dimension reduction methods we consider
5. finally, we also have Ripser [2] and Ripser-image [3] here, upon which all of the persistence analysis is *actually* built

## Structure of the repository

The subdirectories of this repository are listed below, grouped approximately by their role.

### Source Code
Centrally houses code base for project; data cleaning & featurization, calculation, visualization, and key scripting functions are found here.
#### Python repositorty: `src_py`
- `interval-matching_bootstrap` (separate package, see [documentation](https://github.com/tyo8/interval-matching_bootstrap))
- `calculate`:
- `figures`:
- `diagram_distances.py`:
- `collate_tagged_data.py`:
- `HCP_utils.py`:
- `toy_models.py`:
- `lindecomp`:
#### Bash repository: `src_bash`
- `phom_bootstraps.sh`
- `match_bootstraps.sh`
- `submit_*.sh`

Experiments
- 'datalists': lists of data files corresponding that help set conditions for specific experiments
- 'phom_analysis': all persistent homology-based analyses in this project (with the exception of 'metric_tests' and 'bootstrap_benchmarks')
- 'phom_analysis/full-scale-expmt': scripts, code, and tree structure corresponding to investigation in [*paper*] and corresponding figures
- 'nonPH_analysis': most notably contains output of CCA analysis of shared information between pairs of brain representations
- 'metric_tests': **preliminary** testing of the impact of metric choice on persistence data
- 'bootstrap_benchmarks': benchmarking the cycle registration bootstrapping scheme to test scalability
- 'output_benchmarks': becnhmarking the computation(al scaling) of Betti curves, Wasserstein distances, and other derivatives of persistence diagrams

Brain Representation Computation/Extraction
Note that no subject data of any kind is included in this public repository! Instead, the following directories contain the extraction/computation/processing code used to standardize brain representations for persistent homology analysis.
- profumo_reps: code for the computing FC network matrices (correlations between timecourses) and spatial correlation matrices (correlations between maps)
- gradient_reps: code for the computing diffusion-network based gradient representations from connectivity data at both the subject and group level and derivative features of these representations
- ICA_reps: code for computing FC network matrices, partial FC matrices, and amplitude features from extracted ICA-DR data
- glasser: code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from extracted Glasser-parcellated data
- schaefer: code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from Schaefer-parcellated data
- yeo: code for extracting parcellation-level timeseries from data and computing FC network matrices, partial FC matrices, and amplitude features from Yeo-parcellated data

## Preparations

### Compiling the C++ programs (required for cycle registration)
Before running the code to perform cycle matching [1] in this repository, one needs to compile the C++ files in the `modified ripser` folder. For that
- Install a C++ compiler in your computer. We recommend getting the compiler [GCC](https://gcc.gnu.org/).
- The relevant Makefiles are included in the corresponding folders, so the compilation can be done by running the command line `make` in a terminal opened in the folder. 
- The compiled files should be in the same directory than the python scripts/notebooks in which the cycle matching [1] code is invoked.

### Python dependencies
This repo leverages several Python packages. These dependencies are listed below; when only a small number of scripts/modules in the repository use a given package, they are listed beneath the dependency.

#### Parsing Neuroimaging Data
[nibabel](https://nipy.org/nibabel/)
[nilearn](https://nilearn.github.io/stable/index.html)
	
#### Canonical Correlation Analysis (regularized)
[rcca](https://github.com/gallantlab/pyrcca)
	
#### Persistent Homology
[interval-matching_bootstrap](https://github.com/tyo8/interval-matching_bootstrap) [4]
[giotto-tda](https://giotto-ai.github.io/gtda-docs/0.5.1/library.html) [5]
	
#### Figures and Statistics
[python optimal transport library](https://pythonot.github.io/index.html)
[pyRiemann](???)
[scikit-palm](???)
	
#### Background
[numpy](https://numpy.org/)
[scipy](https://scipy.org/)
[pandas](https://pandas.pydata.org/)
[seaborn](https://seaborn.pydata.org/)
[matplotlib](https://matplotlib.org/stable/index.html)
[scikit-learn](https://scikit-learn.org/stable/)

## Academic use

This code is available and is fully adaptable for individual user customization. If you use the our methods, please cite as the following:



```tex
@misc{' ',
	title = {Unknown},
	publisher = {publisher?},
	author = ###############
	month = ??,
	year = {2023/4},
	note = {arXiv:####.<subj> [tag1, tag2]},
	keywords = {topological data analysis, neuroscience, computational topology, persistent homology, functional connectivity, dimension reduction},
}
```

## References
[1] Reani, Yohai, and Omer Bobrowski. 2021. ‘Cycle Registration in Persistent Homology with Applications in Topological Bootstrap’, January. https://arxiv.org/abs/2101.00698v1.

[2] Bauer, Ulrich. 2021. ‘Ripser: Efficient Computation of Vietoris-Rips Persistence Barcodes’. Journal of Applied and Computational Topology 5 (3): 391–423. https://doi.org/10.1007/s41468-021-00071-5.

[3] Bauer, Ulrich, and Maximilian Schmahl. 2022. ‘Efficient Computation of Image Persistence’. ArXiv:2201.04170 [Cs, Math], January. http://arxiv.org/abs/2201.04170.

[4] I. García-Redondo, A. Monod, and A. Song, “Fast Topological Signal Identification and Persistent Cohomological Cycle Matching.” arXiv, Sep. 30, 2022. doi: 10.48550/arXiv.2209.15446. https://arxiv.org/abs/2209.15446

[5] G. Tauzin et al., “giotto-tda: A Topological Data Analysis Toolkit for Machine Learning and Data Exploration,” in NeurIPS 2020 Workshop TDA and Beyond, 2020. Accessed: Mar. 21, 2021. [Online]. Available: https://github.com/giotto-ai/pyflagser.
