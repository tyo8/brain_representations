#!/bin/sh

set -o nounset

ldmZ_fpath=""
data_label="test"
do_img=1
mem_gb=700
partition="small"
maxtime_str="23:55:00"
dim=2
ripser_fpath="/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/modified_ripser/ripser-image-persistence-simple/ripser-image"
script_dir="/scratch/tyoeasley/brain_representations/bootstrap_benchmarks"

while getopts ":x:z:f:o:d:i:m:p:t:D:r:s:" opt; do
  case $opt in
    x) ldmX_fpath=${OPTARG}
    ;;
    z) ldmZ_fpath=${OPTARG}
    ;;
    f) sbatch_fpath=${OPTARG}
    ;;
    o) outpath=${OPTARG}
    ;;
    d) data_label=${OPTARG}
    ;;
    i) do_img=${OPTARG}
    ;;
    m) mem_gb=${OPTARG}
    ;;
    p) partition=${OPTARG}
    ;;
    t) maxtime_str=${OPTARG}
    ;;
    D) dim=${OPTARG}
    ;;
    r) ripser_fpath=${OPTARG}
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

if (($do_img)); then
	sbatch_fpath="${sbatch_fpath}_img"
fi

echo "\
\
#!/bin/sh

#SBATCH --job-name=${data_label}_shbsph
#SBATCH --output=${script_dir}/logs/shbsphoms_${data_label}.out
#SBATCH --error=${script_dir}/logs/shbsphoms_${data_label}.err
#SBATCH --time=${maxtime_str}
#SBATCH --partition=${partition}
#SBATCH --mem=${mem_gb}gb

dim=${dim}
ripser_fpath=${ripser_fpath}

ldmX_fpath=${ldmX_fpath}
ldmZ_fpath=${ldmZ_fpath}

outpath=${outpath}

\
" > "${sbatch_fpath}"  # Overwrite submission script

# slightly different function calls/parameters depending on if we are doing image persistence
if (($do_img))
then
    if [[ -z ${ldmZ_fpath:+.} ]]
    then
        echo "ldmZ_fpath must be set when doing ripser-image"
        exit 1
    fi
    echo "\
\${ripser_fpath} --dim \${dim} --subfiltration \${ldmX_fpath} \${ldmZ_fpath} >> \${outpath}
    " >> "${sbatch_fpath}"
else
    echo "\
\${ripser_fpath} --dim \${dim} \${ldmX_fpath} >> \${outpath}
    " >> "${sbatch_fpath}"
fi

# Make script executable
chmod +x "${sbatch_fpath}" || { echo "Error changing the script permission!"; exit 1; }

# Submit script
echo "Submitting ${sbatch_fpath} and writing phom data to ${outpath} "
sbatch "${sbatch_fpath}" || { echo "Error submitting jobs!"; exit 1; }
echo ""
