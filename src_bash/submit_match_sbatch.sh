#!/bin/sh

set -o nounset

## bookkeeping paths ###
base_dir="/scratch/tyoeasley/brain_representations"
script_dir="${base_dir}/bootstrap_benchmarks"

phomX_fpath=""
phomY_fpath=""
match_homdim=1
sbatch_fpath=""
data_label=""
mem_gb=10
partition="tier2_cpu"
maxtime_str="23:55:00"

while getopts ":x:y:D:f:d:m:p:t:s:" opt; do
  case $opt in
    x) phomX_fpath=${OPTARG}
    ;;
    y) phomY_fpath=${OPTARG}
    ;;
    D) match_homdim=${OPTARG}
    ;;
    f) sbatch_fpath=${OPTARG}
    ;;
    d) data_label=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    s) script_dir=${OPTARG}
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

### paths to code ###
match_script="${base_dir}/src_py/interval-matching-precomp_metric/match/utils_PH/match_script.py"

echo "\
\
#!/bin/sh

#SBATCH --job-name=${data_label}_match
#SBATCH --output=${script_dir}/logs/match_${data_label}.out
#SBATCH --error=${script_dir}/logs/match_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --mem=${mem_gb}gb

match_script=${match_script}

phomX_fpath=${phomX_fpath}
phomY_fpath=${phomY_fpath}
dim=${match_homdim}


python \${match_script} -x \${phomX_fpath} -y \${phomY_fpath} --dim \${dim} -M -d
\
" > "${sbatch_fpath}"  # Overwrite submission script

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
sbatch "${sbatch_fpath}" || { echo "Error submitting job!"; exit 1; }
