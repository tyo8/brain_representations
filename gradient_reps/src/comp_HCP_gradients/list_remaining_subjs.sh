#!/bin/sh

# checks for failed or missing elements; compiles them into a small number of lists

parent_list=${1:-/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv}
output_dir=${2:-/scratch/tyoeasley/brain_representations/gradient_reps/src/comp_HCP_gradients}
img_dir=${3:-/scratch/tyoeasley/brain_representations/gradient_reps/gradmaps_d50}

find ${img_dir} -name emb_* | cut -d- -f2| cut -d. -f1 | sort > ${output_dir}/subjs_done0.csv

# filter done list to contain only subject ID numbers
comm -12 ${parent_list} ${output_dir}/subjs_done0.csv > ${output_dir}/subjs_done.csv
rm ${output_dir}/subjs_done0.csv

# save out ID numbers of non-completed subject embeddings
comm -23 ${parent_list} ${output_dir}/subjs_done.csv > ${output_dir}/subj_IDs_remaining.csv
