base_dir=${1:-"."}
src_path=${2:-"./make_fig.py"}

parent_dirs=$( ls ${base_dir}/*_*/phom_data* -d )

for parent_dir in ${parent_dirs}
do
	echo "Plotting weighted PDs in ${parent_dir}..."
	modality_name=$( basename $( dirname $parent_dir | cut -d_ -f 1 ))
	distance_name=$( basename $parent_dir | cut -d_ -f 5 )
	label="${modality_name}_${distance_name}"
	python ${src_path} -b ${parent_dir}/bars_X.txt -p ${parent_dir}/prevalence_scores_dim1_n250.txt -f ${parent_dir}/weighted_persistence_dgm.png -l ${label}
done
