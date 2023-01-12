in_parent_dir=${1:-/scratch/tyoeasley/brain_representations/schaefer/ext_ptseries/wb_cp}
subj_ids=/scratch/tyoeasley/brain_representations/HCP_IDs_all.csv

dim_dirs=$(ls -d ${in_parent_dir}/d*)

for i in $dim_dirs
do
	out_dir=${i/ext_ptseries/timeseries}
	mkdir $out_dir

	for j in $(cat $subj_ids)
	do
		ts_fname=$out_dir/subj-$j-cat.csv
		cat $(ls $i/*$j*.csv) > $ts_fname
	done
done

python /scratch/tyoeasley/brain_representations/src_py/comp_amps.py $in_parent_dir ${in_parent_dir/ext_ptseries/Amplitudes} True
