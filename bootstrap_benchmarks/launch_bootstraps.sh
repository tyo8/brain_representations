#!/bin/bash

# script directory
script_dir="/scratch/tyoeasley/brain_representations/bootstrap_benchmarks"

distlists_fpath=${1:-${script_dir}"/fast_distlists.csv"}
echo "Running sequential bootstrap benchmarking on: "$distlists_fpath
distlists=$(cat ${distlists_fpath})

# source directory
src_fpath="/scratch/tyoeasley/brain_representations/bootstrap_benchmarks/bootstrap_from_dist_mtx.py"
subsamps=2
n_nodes=1
n_workers=1

### exclude nodes line: maybe do not include for parallel case?
# #SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07
########################## Write the input and the script #########################

for distlist in ${distlists};
do
    for distname in $(cat $distlist)
    do
	dataname=$(basename ${distname} | cut -d. -f 1)
	# Create scripts
	curr_script_file=$(dirname ${distname})"/do_phoms_"${dataname}
	outdir=$(dirname ${distname})"/phoms_"${dataname}
	scratchdir=${outdir}/dist_mtxs
	if ! [ -d $outdir ]; then
		mkdir $outdir
	fi
	if ! [ -d $scratchdir ]; then
		mkdir $scratchdir
	fi
	echo "\
\
#!/bin/sh

#SBATCH --job-name=${dataname}_bsph
#SBATCH --output=${script_dir}/logs/bsphoms_${dataname}.out
#SBATCH --error=${script_dir}/logs/bsphoms_${dataname}.err
#SBATCH --time=7:55:00
#SBATCH --nodes=${n_nodes}
#SBATCH --tasks-per-node=${n_workers}
#SBATCH --mem=25gb

dist_name=${distname}
outdir=${outdir}
subsamps=${subsamps}
n_threads=$(($n_workers * $n_nodes))

cd ${scratchdir}

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top
python ${src_fpath} \${dist_name} -o \${outdir} -s \${subsamps} -n ${n_workers} --nopar
\
	" > "${curr_script_file}"  # Overwrite submission script

	# Make script executable
	chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    	# Submit script
    	sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
    done
done
