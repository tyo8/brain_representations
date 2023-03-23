#!/bin/sh

src_data_spec="tyoeasley@login3.chpc.wustl.edu:/scratch/tyoeasley/brain_representations"
target_dir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations"

rsync -rv ${src_data_spec}/* ${target_dir} --max-size=26K --exclude=*phom_data* --exclude=*.txt --exclude=*.nii.gz* --exclude=logs --exclude=*sub-* --exclude=*subj-*.csv* --exclude=*.html --exclude=*.hdf5 --exclude=*.m --exclude=*MATLAB* --exclude=*dist_mtx* --exclude=*data.csv --exclude=*.pfm --exclude=HCP_PTN1200 --exclude=*deprecated*
