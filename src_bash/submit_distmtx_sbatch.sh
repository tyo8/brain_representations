#!/bin/bash

set -o nounset

# script directory
base_dir="/scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within-PROFUMO"

modality=ICA
feature=Amps

mem_gb=500
partition="tier2_cpu"
maxtime_str="167:55:00"

while getopts ":b:m:f:g:p:t:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    m) modality=${OPTARG}
    ;;
    f) feature=${OPTARG}
    ;;
    g) mem_gp=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

script_dir="${base_dir}/${modality}_${feature}"
methods_fpath=${script_dir}"/methods.csv"
methods=$(cat ${methods_fpath})
# path to source file
pysrc_fpath="/scratch/tyoeasley/brain_representations/src_py/comp_sim_mtx.py"



########################## Write the input and the script #########################

for method in ${methods};
do
    echo $method
    # Create scripts
    sbatch_fname=${script_dir}"/do_sim_"${feature}${modality}${method}
    label=dists		# was previously either "sims" or "dists" but "sims" use case is deprecated
    echo "\
\
#!/bin/sh

#SBATCH --job-name=${modality}${feature}_${method}
#SBATCH --output=${base_dir}/logs/${modality}${feature}_${method}.out
#SBATCH --error=${base_dir}/logs/${modality}${feature}_${method}.err
#SBATCH --partition=${partition}
#SBATCH --time=${maxtime_str}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=${mem_gb}gb

subj_list=${script_dir}/subj_list_${feature}.csv
subj_dist=${script_dir}/${modality}_${feature}_${method}_${label}.txt

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro
python ${pysrc_fpath} \${subj_list} \${subj_dist} ${method} True

echo \${subj_dist} >> ${script_dir}/${feature}${modality}_${label}.csv
\
" > "${sbatch_fname}"  # Overwrite submission script

    # Make script executable
    chmod +x "${sbatch_fname}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${sbatch_fname}" || { echo "Error submitting jobs!"; exit 1; }
done
