#!/bin/bash

# number of components (coordinate dimensions) in gradient embedding
n_comps=50

# Filepath to list of subject ID numbers
subjID_fpath="/scratch/tyoeasley/brain_representations/gradient_reps/src/comp_HCP_gradients/subj_IDs_remaining.csv"
# subjID_fpath="/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv"
# subjID_fpath="/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_tst.csv"
subjID_list=$(cat ${subjID_fpath})

# Script directory
script_dir="/scratch/tyoeasley/brain_representations/gradient_reps/src/comp_HCP_gradients"
curr_script_file=${script_dir}"/do_comp_subj_grad"

# Log directory
log_dir="/scratch/janine.bijsterbosch/WAPIAW_2/heterogeneity/log"  # Default log directory name ("launch-001" to "launch-999")




########################## Write the input and the script #########################

for subjID in ${subjID_list};
do
    # Create scripts
    SLURM_out=${script_dir}"/logs/subj-logs_d${n_comps}/subj-"${subjID}"_grad"
    echo "\
\
#!/bin/sh
#SBATCH --job-name=subj_grad
#SBATCH --output=${SLURM_out}.out
#SBATCH --error=${SLURM_out}.err
#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07,node21
#SBATCH --time=1:55:00
#SBATCH --mem=100GB

## Note: much more time & memory resources are required when using the -t/--tanproj flag
#  time: >= 1 week (per subject)
#   mem: >= 270GB (per subject)

src_path=${script_dir}/compute_subj_gradients.py

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro
python \${src_path} ${subjID} --par -n ${n_comps}
\
" > "${curr_script_file}"  # Overwrite submission script

    # Make script executable
    chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
done



