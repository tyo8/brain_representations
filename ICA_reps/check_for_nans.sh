feat_types=("Amplitudes" "NetMats" "partial_NMs" "node_timeseries")
dimset=(15 25 50 100 200 300)

basedir="/scratch/tyoeasley/brain_representations/ICA_reps"
py_src="/scratch/tyoeasley/brain_representations/src_py/check_for_nans.py"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro

for feat in ${feat_types[@]}
do
	echo "feature type: ${feat}"
	for dim in ${dimset[@]}
	do
		echo "observing data from dimension ${dim} extraction: data directory is"
		echo "${py_src} ${basedir}/3T_HCP1200_MSMAll_d${dim}_ts2_Z/${feat}"
		python ${py_src} "${basedir}/3T_HCP1200_MSMAll_d${dim}_ts2_Z/${feat}"
		echo ""
	done
done
