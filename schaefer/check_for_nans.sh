feat_types=("Amplitudes" "NetMats" "partial_NMs" "timeseries")
dimset=(100 200 300 600 1000)

basedir="/scratch/tyoeasley/brain_representations/schaefer"
py_src="/scratch/tyoeasley/brain_representations/src_py/check_for_nans.py"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro

for feat in ${feat_types[@]}
do
	echo "feature type: ${feat}"
	for dim in ${dimset[@]}
	do
		echo "observing data from dimension ${dim} extraction: data directory is"
		echo "${basedir}/${feat}/d${dim}"
		python ${py_src} ${basedir}/${feat}/d${dim}
		echo ""
	done
done
