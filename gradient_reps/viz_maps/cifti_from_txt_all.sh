#!/bin/bash

load_files=$(find . -name '*.txt')
template_file=/home/tyo/Documents/Personomics_Research_Group/Experiments/Archives/NMFvsICA/CIFTI_out/uniform_spmaps_CIFTI/NMF100_umap_gray_normvar.dtseries.nii


for fin in $load_files
do
	fout=${fin/txt/dtseries.nii}
	wb_command -cifti-convert -from-text $fin $template_file $fout
done
