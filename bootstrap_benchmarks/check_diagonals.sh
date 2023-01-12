#!/bin/bash

# script directory
script_dir="/scratch/tyoeasley/brain_representations/bootstrap_benchmarks"

distlists_fpath=${1:-${script_dir}"/real_distlists.csv"}
echo "Checking diagonals of distance matrices listed in "$distlists_fpath
distlists=$(cat ${distlists_fpath})


for distlist in ${distlists};
do
    for distname in $(cat $distlist)
    do
	python shapecheck.py $distname
    done
done
