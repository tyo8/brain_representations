#!/bin/sh

set -o nounset

## bookkeeping paths ###
base_dir="interval-matching_pairperms"
outdir="${base_dir}/phom_analysis/pairperms_distances/PROFUMO_vs_null"

barX_fpath=""
barY_fpath=""
permtype="subject"
count=100
pairperms_homdim=1
p=2
q=2
use_affinity=true
verbose=true

mem_gb=10
partition="tier2_cpu"
maxtime_str="23:55:00"

while getopts ":b:x:y:T:n:D:P:Q:a:v:o:m:p:t:" opt; do
  case $opt in
    b) base_dir=${OPTARG}
    ;;
    x) barX_fpath=${OPTARG}
    ;;
    y) barY_fpath=${OPTARG}
    ;;
    T) permtype=${OPTARG}
    ;;
    n) count=${OPTARG}
    ;;
    D) pairperms_homdim=${OPTARG}
    ;;
    P) p=${OPTARG}
    ;;
    Q) q=${OPTARG}
    ;;
    a) use_affinity=${OPTARG}
    ;;
    v) verbose=${OPTARG}
    ;;
    o) outdir=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
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

### paths to code ###
pairperms_script="${base_dir}/src_py/calculate/comp_pair_permtest_dists.py"
Xname=$(basename $(dirname $(ls ${barX_fpath})))
Xname=${Xname/phom_data_/}
Xname=${Xname/_dists/}
Yname=$(basename $(dirname $(ls ${barY_fpath})))
Yname=${Yname/phom_data_/}
Yname=${Yname/_dists/}
data_label="${Xname}-vs-${Yname}"

sbatch_fpath="${outdir}/do_nullpairdists_${data_label}"

echo "\
\
#!/bin/sh

#SBATCH --job-name=pairperms_${data_label}
#SBATCH --output=${outdir}/logs/pairperms_${data_label}.out
#SBATCH --error=${outdir}/logs/pairperms_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem=${mem_gb}gb

pairperms_script=${pairperms_script}

barX_fpath=${barX_fpath}
barY_fpath=${barY_fpath}
permtype=${permtype}
dim=${pairperms_homdim}
outdir=${outdir}

echo \"barX_fpath: \\\"\${barX_fpath}\\\"\"
echo \"barY_fpath: \\\"\${barY_fpath}\\\"\"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate stats
echo \"saving results to \${outdir}\"

python3 \${pairperms_script} -x \${barX_fpath} -y \${barY_fpath} -t \${permtype} --dim \${dim} -p ${p} -q ${q} -o \${outdir} -v
\
" > "${sbatch_fpath}"  # Overwrite submission script

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
sbatch "${sbatch_fpath}" || { echo "Error submitting job!"; exit 1; } 
