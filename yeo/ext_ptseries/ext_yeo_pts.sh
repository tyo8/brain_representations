#!/bin/sh

#	HCP_path=/scratch/janine.bijsterbosch/HCP/%s/3T/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii%.s%.s
#	subj_list=/scratch/tyoeasley/brain_representations/schaefer/ext_ptseries/HCP_tst_subjs.csv
HCP_path=/ceph/hcpdb/archive/HCP_1200/arc001/%s_3T/RESOURCES/MSMAllDeDrift/MNINonLinear/Results/rfMRI_REST%s_%s/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii
subj_list=/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv

yeo_path=/scratch/tyoeasley/brain_representations/yeo/parcellation_defs/Yeo2011_17Networks_N1000.dlabel.nii

ext_type=${1:-wb_cp}
out_dir=/scratch/tyoeasley/brain_representations/yeo/ext_ptseries/${ext_type}
out_path=${out_dir}"/%s_REST%s_%s.ptseries.nii"


if [ ! -d ${out_dir} ]; then
	mkdir -p ${out_dir}
	echo "WARNING: created directory "${out_dir}
fi

echo "Timeseries data extraction method: "${ext_type}
echo "Extracting from: "${HCP_path}
echo "Extracting to: "${out_dir}
echo "extracting..."

case $ext_type in
	wb_cp)
		module load workbench
		for s in `cat ${subj_list}` ; do
			HCP_path_s=$(/scratch/tyoeasley/brain_representations/check_subj.sh ${s} ${HCP_path})
			wb_command -cifti-parcellate $(printf ${HCP_path_s} ${s} "1" "LR" "1" "LR") ${yeo_path} COLUMN $(printf ${out_path} ${s} "1" "LR") -legacy-mode
			wb_command -cifti-parcellate $(printf ${HCP_path_s} ${s} "1" "RL" "1" "RL") ${yeo_path} COLUMN $(printf ${out_path} ${s} "1" "RL") -legacy-mode
			wb_command -cifti-parcellate $(printf ${HCP_path_s} ${s} "2" "LR" "2" "LR") ${yeo_path} COLUMN $(printf ${out_path} ${s} "2" "LR") -legacy-mode
			wb_command -cifti-parcellate $(printf ${HCP_path_s} ${s} "2" "RL" "2" "RL") ${yeo_path} COLUMN $(printf ${out_path} ${s} "2" "RL") -legacy-mode
		done

		source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top_neuro
		python /scratch/tyoeasley/brain_representations/src_py/convert_to_csv.py ${out_dir} .ptseries.nii
		;;
	

	MATLAB)
		out_path=${out_path/.ptseries.nii/.csv}
		cwd=$PWD
		cd /scratch/tyoeasley/brain_representations/src_MATLAB
		module load matlab
		matlab -nodisplay -nojvm -batch "ext_ptseries(\"${HCP_path}\",\"${yeo_path}\",\"${out_path}\",\"${subj_list}\")"
		cd ${cwd}
		;;


	NiBabel)
		source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top_neuro
		python ???????.py ${parcel_num} ${subj_list} ${out_dir} ${HCP_path}
		;;


#	Nilearn)
#		echo "extracting..."
#		source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate alg_top_neuro
#		python ext_nilearn_schaefer.py ${parcel_num} ${subj_list} ${out_dir} ${HCP_path}
#		;;
esac

echo "done."
