dX_pattern=${1:-"L2scale.txt"}
dir_pattern=${2:-"within_*"}
base=${3:-$(pwd)}

echo ""
echo "Matching \"*${dX_pattern}\" in directories of type \"${dir_pattern}\""
echo ""
for i in $(ls ${base}/${dir_pattern} -d)
do
	distlist_out="${i}/distlist_${dX_pattern/txt/csv}"
	ls ${i}/*${dX_pattern} > ${distlist_out}
	echo "Matches with ${i} collected in:"
	ls "${distlist_out}" -lh
	
	distlist_set="${i}/distlist_set_${dX_pattern/txt/csv}"
	echo ${distlist_out} >> ${distlist_set} 
	echo "Collections of matches aggregated in:"
	ls "${distlist_set}" -lh
	sort -o ${distlist_set} -u ${distlist_set} 	# removes duplicate lines from distlist_set
done
