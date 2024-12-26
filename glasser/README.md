# Diffusion Gradient Brain Representations 
Computes the dense (unparcellated) connectivity matrix from subject data, which then defines the kernel of a diffusion operator used to compute a parallel transport distance between temporal states of rfMRI timeseries data; these states are then projected back onto the cortical surface.

## References
The diffusion gradient brain representation approach is derived from the following work:
- [1] R. R. Coifman and S. Lafon, “Diffusion maps,” Applied and Computational Harmonic Analysis, vol. 21, no. 1, pp. 5–30, Jul. 2006, doi: 10.1016/j.acha.2006.04.006.

- [2] Langs, Golland and Ghosh (2015) Predicting Activation Across Individuals with Resting-State Functional Connectivity based Multi-Atlas Label Fusion. MICCAI. 

- [3] D. S. Margulies et al., “Situating the default-mode network along a principal gradient of macroscale cortical organization,” Proceedings of the National Academy of Sciences, vol. 113, no. 44, pp. 12574–12579, Nov. 2016, doi: 10.1073/pnas.1608282113.

## Python dependencies
The following packages are necessary for computing diffusion gradient decompositions of Human Connectome Project (HCP-YA) data using the code in `src`: 
- [hd5py](https://docs.h5py.org/en/stable/)
- [nibabel](https://nipy.org/nibabel/)
- [mapalign](https://github.com/sensein/mapalign)
- [scikit-learn](https://scikit-learn.org/stable/)
