#!/bin/bash

submit=${1:-false}
dX_pattern=${2:-"*_dists.txt"}
dir_pattern=${3:-"within_*"}
subdir_pattern=${4:-"*_*"}
base=${5:-$(pwd)}

echo ""
echo "Matching \"${dX_pattern}\" in directories of type \"${dir_pattern}/${subdir_pattern}\""
echo ""
for j in $(ls ${base}/${dir_pattern} -d) 
do
	distlist_set="${j}/distlist_set.csv"

	for i in $( ls ${j}/${subdir_pattern} -d)
	do
			echo ""
			distlist_out="${i}/distlist.csv"
			echo "Found $(ls ${i}/${dX_pattern} | wc -l) matches" 
			ls ${i}/${dX_pattern} > ${distlist_out} || { echo "no matches found; deleting and moving on."; rm ${distlist_out}; continue; exit 1; }
			echo "Collected $(wc -l ${distlist_out} | cut -d' ' -f 1) matches with ${i} in:"
			ls "${distlist_out}"
			echo "first line: $(ls $(cat ${distlist_out} | head -1 ))"
			echo "${distlist_out}" >> ${distlist_set} 
	done

	echo "Collections of matches aggregated in:"
	ls "${distlist_set}"
	sort -o ${distlist_set} -u ${distlist_set} 	# removes duplicate lines from distlist_set
	echo "first line (of $(cat ${distlist_set} | wc -l) in collection): $( ls $( cat ${distlist_set} | head -1 ))"
	echo ""

	if $submit
	then
		cd ${j}
		echo "${j}"
		sbatch "${j}/do_bash_phom_bootstraps"
	fi
done
