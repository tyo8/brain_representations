#!/bin/sh

set -o nounset

## bookkeeping paths ###
# base_dir="interval-matching_bootstrap"
xtr_src="interval-matching_bootstrap/utils_match/extract.py"

phomX_fpath=""
sbatch_fpath=""
data_label=""
mem_gb=2
partition="tier2_cpu"
maxtime_str="23:55:00"

while getopts ":x:f:d:s:m:p:t:" opt; do
  case $opt in
    x) phomX_fpath=${OPTARG}
    ;;
    f) sbatch_fpath=${OPTARG}
    ;;
    d) data_label=${OPTARG}
    ;;
    s) xtr_src=${OPTARG}
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
logdir=$( dirname ${sbatch_fpath} )

echo "\
\
#!/bin/sh

#SBATCH --job-name=${data_label}_xtrbars
#SBATCH --output=${logdir}/logs/xtrbars_${data_label}.out
#SBATCH --error=${logdir}/logs/xtrbars_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem=${mem_gb}gb

xtr_src=${xtr_src}

phomX_fpath=${phomX_fpath}

echo \"phomX_fpath: \\\"\${phomX_fpath}\\\"\"

python3 \${xtr_src} -x \${phomX_fpath} -0 -w -v
echo "xtr_srced to:"
ls \${phomX_fpath/phom_X/bars_X}
\
" > "${sbatch_fpath}"  # Overwrite submission script

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
sbatch "${sbatch_fpath}" || { echo "Error submitting job!"; exit 1; }
