in_parent_dir=${1:-/scratch/tyoeasley/brain_representations/yeo/ext_ptseries/wb_cp}
subj_ids=/scratch/tyoeasley/brain_representations/HCP_IDs_all.csv

out_dir=${in_parent_dir/ext_ptseries/timeseries}
mkdir $out_dir

for j in $(cat $subj_ids)
do
	ts_fname=$out_dir/subj-$j-cat.csv
	cat $(ls $i/*$j*.csv) > $ts_fname
done

python /scratch/tyoeasley/brain_representations/src_py/comp_amps.py $in_parent_dir ${in_parent_dir/ext_ptseries/Amplitudes} False
