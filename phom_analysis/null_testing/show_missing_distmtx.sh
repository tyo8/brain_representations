#!/bin/bash

get_subname() {
	local fname=$1
	local rep=$(echo $fname | cut -d_ -f 1)
	local feat=$(echo $fname | cut -d_ -f 2)
	local dist=$(echo $fname | cut -d_ -f 3)
	local newname=${fname/${rep}/do}
	local newname=${newname/${feat}/sim}
	local newname=${newname/${dist}/${feat}${rep}${dist}}
	local newname=${newname/dists_/}
	local subname=${newname/.txt/}
	echo ${subname}
}

maxnum=999
submit=false
permprefix="perm_set"
dX_pattern="*method*Perms*.txt"
dir_pattern="within_*"
subdir_pattern="*_*/permstrapping"
base_dir="/scratch/tyoeasley/brain_representations/phom_analysis/null_testing"
verbose=true

### argument parsing ###
while getopts ":n:s:p:x:D:d:b:v:" opt; do
  case $opt in
    n) maxnum=${OPTARG}
    ;;
    s) submit=${OPTARG}
    ;;
    p) permprefix=${OPTARG}
    ;;
    x) dX_pattern=${OPTARG}
    ;;
    D) dir_pattern=${OPTARG}
    ;;
    d) subdir_pattern=${OPTARG}
    ;;
    D) maxhomdim=${OPTARG}
    ;;
    b) base_dir=${OPTARG}
    ;;
    v) verbose=${OPTARG}
    ;;
    \?) echo "Invalid option -${OPTARG}" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

dX_pattern="${dX_pattern/.txt/${permprefix}*.txt}"


echo "Matching \"${dX_pattern}\" in directories of type \"${dir_pattern}/${subdir_pattern}\""
echo ""
for j in $(ls ${base_dir}/${dir_pattern} -d) 
do
	echo "Searching for missing output files in ${j}..."
	for i in $( ls ${j}/${subdir_pattern} -d)
	do
		methods=$( cat $( dirname ${i} )/methods.csv )
		echo "subdir: ${i}"
		for method in ${methods}
		do
			dX_pattern=${dX_pattern/method/$method}
			matchnum=$(ls ${i}/${dX_pattern} | wc -l) || { echo "no matches found to '${dX_pattern}'; deleting and moving on."; rm ${distlist_out}; continue; exit 1; }
			top_match=$(ls ${i}/${dX_pattern} | head -1)
			#`echo "First match looks like: ${top_match}"
			if [[ "${matchnum}" -lt "$(( ${maxnum}+1 ))" ]]
			then
				if $verbose
				then
					echo "Found ${matchnum} matches to '${dX_pattern}'; expected $(( ${maxnum}+1 ))."
				fi
				for n in $( seq 0 $maxnum )
				do
					fpath=${top_match/${permprefix}0/${permprefix}${n}}
					if ! test -f ${fpath}
					then
						subname=$( get_subname $(basename ${fpath}) )
						if $verbose
						then
							echo "Not found: ${fpath}"
							echo "has corresponding submission script:"
						fi
						ls ${i}/${subname}
						if $submit
						then
							cd ${i}
							echo "submitting ${subname}"
							sbatch ${i}/${subname}
							cd ${base_dir}
							echo ""
						fi
						if $verbose
						then
							echo ""
						fi
					fi
				done
				if $verbose
				then
					echo ""
				fi
			fi
			dX_pattern=${dX_pattern/${method}/method}
		done
	done
	echo "done."
	echo ""
done


