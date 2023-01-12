#!/bin/sh

ext_type=${1:-wb_cp}
dims=$(<dims.txt)
fname="do_ext_schaefer_ptseries"

for i in ${dims}
do
	echo "#!/bin/sh" > ${fname}
	echo "#SBATCH --job-name=ext_dts_schaefer" >> ${fname}
	echo "#SBATCH --output=/scratch/tyoeasley/brain_representations/schaefer/ext_ptseries/logs/ext_dts_%j.out" >> ${fname}
	echo "#SBATCH --error=/scratch/tyoeasley/brain_representations/schaefer/ext_ptseries/logs/ext_dts_%j.err" >> ${fname}
	echo "#SBATCH --time=23:55:00" >> ${fname}
	echo "#SBATCH --mem=29GB" >> ${fname}
	echo "" >> ${fname}
	echo "" >> ${fname}
	echo "module load workbench" >> ${fname}
	echo "" >> ${fname}
	echo "bash /scratch/tyoeasley/brain_representations/schaefer/ext_ptseries/ext_schaefer_dts.sh "${i}" "${ext_type} >> ${fname}

	sbatch ${fname}
done
