#!/bin/sh

report_file=/scratch/tyoeasley/brain_representations/hcpdb_data_report.txt

subj_path_type=${2:-/ceph/hcpdb/archive/HCP_1200/arc001/%s_3T/RESOURCES/MSMAllDeDrift/MNINonLinear/Results/rfMRI_REST%s_%s/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii}

subj_path=$(printf ${subj_path_type} $1 "1" "LR" "1" "LR")
subj_path=${subj_path/HCP_500/HCP_1200}
subj_path=${subj_path/HCP_900/HCP_1200}

if test -s $subj_path
then
	subj_path_type=${subj_path_type}
	# echo "Subject "$1" found in HCP_1200 release" >> ${report_file}
elif test -s ${subj_path/HCP_1200/HCP_900}
then
	# echo "Subject "$1" found in HCP_900 release" >> ${report_file}
	subj_path_type=${subj_path_type/HCP_1200/HCP_900}
elif test -s ${subj_path/HCP_1200/HCP_500}
then
	# echo "Subject "$1" found in HCP_500 release" >> ${report_file}
	subj_path_type=${subj_path_type/HCP_1200/HCP_500}
else
	subj_path_type=$1"_SUBJ_NOT_FOUND"
fi

subj_path=$(printf ${subj_path_type} $1 "1" "LR" "1" "LR")
echo $subj_path
