modality=${1:-PROFUMO}
fname_suffix=${2:-Maps}

dirbase=/scratch/tyoeasley/brain_representations

case $modality in

	PROFUMO)
		dirname=${dirbase}/profumo_reps/HCP_Profumo_cifti_bigdata_1200subs_50modes_smooth_5mm_4runs.ppp
		ext=%s
		;;
	
	ICA)
		dim=$(expr modality : "[^0-9]*\([0-9]*\)")
		dirname=${dirbase}/ICA_reps/3T_HCP1200_MSMAll_d${dim}_ts2_Z
		ext=%s
		;;

	schaefer)
		dim=$(expr modality : "[^0-9]*\([0-9]*\)")
		dirname=${dirbase}/schaefer
		ext=%s/d${dim}
		;;
	MMP)
		dirname=${dirbase}/glasser
		ext=%s
		;;
	yeo)
		dirname=${dirbase}/yeo
		ext=%s
		;;
	grad)
		dim=$(expr modality : "[^0-9]*\([0-9]*\)")
		dirname=${dirbase}/gradient_reps
		ext=%s/d${dim}
		;;
esac

case $fname_suffix in

	NM)
		ext=$(printf ${ext} /NetMats)
		;;

	Maps)
		ext=$(printf ${ext} /Maps)
		;;

	Amps)
		ext=$(printf ${ext} /Amplitudes)
		;;

	pNM)
		ext=$(printf ${ext} /partial_NMs)
		;;
esac

dirname=${dirname}${ext}

if ! [ -d ${dirname}/logs ]; then
	mkdir ${dirname}/logs
fi

filename=${dirname}/do_pSim_${fname_suffix}

echo "#!/bin/sh" > ${filename}
printf "\n" >> ${filename}
echo "#SBATCH --job-name=psim"${modality}_${fname_suffix} >> ${filename}
echo "#SBATCH --output="${dirname}"/logs/psim"_${fname_suffix}"%j.out" >> ${filename}
echo "#SBATCH --error="${dirname}"/logs/psim"_${fname_suffix}"%j.err" >> ${filename}
echo "#SBATCH --time=147:55:00" >> ${filename}
echo "#SBATCH --nodes=1" >> ${filename}
echo "#SBATCH --tasks-per-node=1" >> ${filename}
echo "#SBATCH --mem=50gb" >> ${filename}
printf "\n" >> ${filename}
printf "\n" >> ${filename}
echo "subj_list="${dirname}"/subj_list"_${fname_suffix}".csv" >> ${filename}
echo "subj_dist="${dirname}"/subj_psims"_${fname_suffix}"_"${modality}".csv" >> ${filename}
printf "\n" >> ${filename}
echo "cd "${dirname} >> ${filename}
printf "\n" >> ${filename}
echo "source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top_neuro" >> ${filename}
printf "python "${dirbase}"/src_py/comp_Psim_mtx.py $" >> ${filename}
printf "{subj_list} $" >> ${filename}
printf "{subj_dist}\n" >> ${filename}
printf ${dirbase}"/add_to_dists.sh $" >> ${filename}
printf "{subj_dist}\n" >> ${filename}
printf "\n" >> ${filename}
