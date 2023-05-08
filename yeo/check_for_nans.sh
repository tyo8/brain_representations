feat_types=("Amplitudes" "NetMats" "partial_NMs" "timeseries")

basedir="/scratch/tyoeasley/brain_representations/yeo"
py_src="/scratch/tyoeasley/brain_representations/src_py/check_for_nans.py"

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate neuro

for feat in ${feat_types[@]}
do
	echo "observing data from dimension ${dim} extraction: data directory is"
	echo "${basedir}/${feat}"
	python ${py_src} "${basedir}/${feat}"
	echo ""
done
