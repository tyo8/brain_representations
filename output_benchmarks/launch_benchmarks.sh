#!/bin/bash

# output type to benchmark
# benchmark_type="bcurves"
benchmark_type="W2dists"

# script directory
script_dir="/scratch/tyoeasley/brain_representations/output_benchmarks"

simlists_fpath=${script_dir}"/all_simlists.csv"
simlists=$(cat ${simlists_fpath})

# source directory
src_fpath="/scratch/tyoeasley/brain_representations/output_benchmarks/benchmark_"${benchmark_type}".py"


########################## Write the input and the script #########################

for simlist in ${simlists};
do
    dataname=$(basename ${simlist} | cut -d. -f 1)
    # Create scripts
    curr_script_file=$(dirname ${simlist})"/do_${benchmark_type}_"${dataname}
    echo "\
\
#!/bin/sh

#SBATCH --exclude=node22,node29,node31,node15,node25,node30,node24,node28,node08,node07,node21
#SBATCH --job-name=${benchmark_type}_${dataname}
#SBATCH --output=${script_dir}/logs/${benchmark_type}_${dataname}.out
#SBATCH --error=${script_dir}/logs/${benchmark_type}_${dataname}.err
#SBATCH --time=3:55:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=100gb

sim_list=${simlist}

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top
python ${src_fpath} \${sim_list}
\
" > "${curr_script_file}"  # Overwrite submission script

    # Make script executable
    chmod +x "${curr_script_file}" || { echo "Error changing the script permission!"; exit 1; }

    # Submit script
    sbatch "${curr_script_file}" || { echo "Error submitting jobs!"; exit 1; }
done



