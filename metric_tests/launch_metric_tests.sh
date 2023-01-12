#!/bin/bash

feature=${1:-Amps}
modality=${2:-ICA300}

# script directory
script_dir="/scratch/tyoeasley/brain_representations/metric_tests/"${modality}_${feature}

methods_fpath=${script_dir}"/methods.csv"
methods=$(cat ${methods_fpath})

# source directory
src_fpath="/scratch/tyoeasley/brain_representations/src_py/comp_sim_mtx.py"


########################## Write the input and the script #########################

for method in ${methods};
do
    echo $method
    # Create scripts
    curr_script_file=${script_dir}"/do_sim_"${feature}${modality}${method}
    if [[ $(echo $method) == 'geodesic' ]]; then
	    label=dists
    else
	    label=sims
    fi
    echo "\
\
#!/bin/sh

#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07
#SBATCH --job-name=${modality}${feature}_${method}
#SBATCH --output=/scratch/tyoeasley/brain_representations/metric_tests/logs/${modality}${feature}_${method}.out
#SBATCH --error=/scratch/tyoeasley/brain_representations/metric_tests/logs/${modality}${feature}_${method}.err
#SBATCH --time=23:55:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=50gb

subj_list=/scratch/tyoeasley/brain_representations/metric_tests/${modality}_${feature}/subj_list_${feature}.csv
subj_sim=/scratch/tyoeasley/brain_representations/metric_tests/${modality}_${feature}/${modality}_${feature}_${method}_${label}.txt

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate Stats
python ${src_fpath} \${subj_list} \${subj_sim} ${method}
if test -f ${subj_sim}
then
	echo \${subj_sim} >> /scratch/tyoeasley/brain_representations/metric_tests/${modality}_${feature}/${feature}${modality}_sims.csv
fi
\
" > "${curr_script_file}"  # Overwrite submission script

    # Make script executable
    chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
done



