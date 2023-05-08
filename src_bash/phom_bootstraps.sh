#!/bin/bash

set -o nounset

### bookkeeping paths ###
base_dir="/scratch/tyoeasley/brain_representations/"
subbase_dir="${base_dir}/bootstrap_benchmarks"
tagpath="${base_dir}/taglist.txt"

### input datasets ###
distlists_fpath="${subbase_dir}/real_distlists.csv"

### run parameters ###
samps=3
bootstrap_prop=0.8
maxhomdim=1
do_X_phom=1
mem_gb=300

### argument parsing ###
while getopts ":b:s:f:t:n:p:D:x:m:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    s) subbase_dir=${OPTARG}
    ;;
    f) distlists_fpath=${OPTARG}
    ;;
    t) tagpath=${OPTARG}
    ;;
    n) samps=${OPTARG}
    ;;
    p) bootstrap_prop=${OPTARG}
    ;;
    D) maxhomdim=${OPTARG}
    ;;
    x) do_X_phom=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
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

### paths to code ###
ripser_img_path="/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/modified_ripser/ripser-image-persistence-simple/ripser-image"
ripser_rep_path="/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/modified_ripser/ripser-tight-representative-cycles/ripser-representatives"
ripser_scripter="${base_dir}/src_bash/submit_ripser_sbatch.sh"
gentags_script="${base_dir}/src_py/generate_subindex.py"
ldm_script="${base_dir}/src_py/interval-matching-precomp_metric/match/utils_PH/ldm_script.py"

### node exclude list: maybe do not include for parallel case? ###
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07

############################### Write the input and the script ##############################

echo "Running bash-anarchic benchmarking on: "${distlists_fpath}
distlists=$(cat ${distlists_fpath})

ndims=1003
if compgen -G $tagpath >> /dev/null
then
	echo "Running ${samps} bootstraps from taglist \"${tagpath}\" with maximum homology dimension ${maxhomdim}."
	tags=$( cat ${tagpath} | head -$samps )
else
	echo "Running ${samps} bootstraps at a sampling proportion of ${bootstrap_prop} with maximum homology dimension ${maxhomdim}."
	tags=$(python ${gentags_script} --count ${samps} --dims ${ndims} --proportion ${bootstrap_prop})
	for tag in $tags
	do
		echo $tag >> $tagpath
	done
fi

echo "Do X phom? (numeric) t/f: ${do_X_phom}"
printf '\n\n'

for distlist in ${distlists}
do
	echo ""
	echo "Pulling from ${distlist}"
    	for distname in $(cat $distlist)
    	do
		echo "Currently bootstrapping data for ${distname}"
		data_label=$(basename ${distname} | cut -d. -f 1)

		# Name the submission script for this dataset type
		sbatch_fpath=$(dirname ${distname})"/do_phoms_${data_label}"

		# Create bookkeeping directories
		outdir=$(dirname ${distname})"/phom_data_${data_label}"
		scratchdir="${outdir}/dist_mtxs"
		mkdir -p $scratchdir
		phomdir="${outdir}/phom_out"
		mkdir -p $phomdir

		dX_fpath="${scratchdir}/dX.ldm"
		cp $distname $dX_fpath
		ndims=$(wc -l ${dX_fpath})

		phomX_outpath="${outdir}/phom_X.txt"
		if ((${do_X_phom})) & ! [ -f ${phomX_outpath} ]
		then
			# submit persistence job for dX
			${ripser_scripter} -x ${dX_fpath} -f ${sbatch_fpath} -o ${phomX_outpath} -r ${ripser_rep_path} -i 0 -m 50 -p "small" -d ${data_label}"_X" -D ${maxhomdim}
		fi

    		for tag in $tags
		do
			# compute dZ_(tag) and dY_(tag) from dX and subsample indices (via `_subsamp_dZ` subfunction of `create_matrices_image.py')
			# make dXZ_(tag) and dYZ_(tag) (via `create_matrices_image.py`)
			dY_fpath=${scratchdir}"/dY_${tag}.ldm"
			dXZ_fpath=${scratchdir}"/dXZ_${tag}.ldm"
			dYZ_fpath=${scratchdir}"/dYZ_${tag}.ldm"
			dZ_fpath=${scratchdir}"/dZ_${tag}.ldm"
			if ! test -f $dZ_fpath
			then
				python ${ldm_script} -x ${dX_fpath} -t "${tag}" -z ${dZ_fpath} -y ${dY_fpath} -i ${dXZ_fpath} -j ${dYZ_fpath} || \
					printf "Failed to parse arguments to ldm_script (likely unreadable tag): \n${tag}\n\n"
			fi

			# name the (generic type of) output path for persistent homology data
			phom_outpath=${phomdir}"/phomXZ_"${tag}".txt"

			# submit image persistence job for dXZ_(tag) and dYZ_(tag) --in style of `do_ripser_test` sbatch script
			if ! [ -f $phom_outpath ]
			then
				${ripser_scripter} -x ${dXZ_fpath} -z ${dZ_fpath} -f ${sbatch_fpath} -o ${phom_outpath} -m ${mem_gb} -p "small" -d ${data_label}"_XZ" -D ${maxhomdim}
			fi
			if ! [ -f ${phom_outpath/phomXZ/phomYZ} ]
			then
				${ripser_scripter} -x ${dYZ_fpath} -z ${dZ_fpath} -f ${sbatch_fpath} -o ${phom_outpath/phomXZ/phomYZ} -m ${mem_gb} -p "small" -d ${data_label}"_YZ" -D ${maxhomdim}
			fi

			# submit persistence job for dY_(tag)
			if ! [ -f ${phom_outpath/phomXZ/phomY} ]
			then
				${ripser_scripter} -x ${dY_fpath} -f ${sbatch_fpath} -o ${phom_outpath/phomXZ/phomY} -r ${ripser_rep_path} -i 0 -d ${data_label}"_Y" -D ${maxhomdim}
			fi
		done
	done
done
