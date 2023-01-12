new_fname=${1}

list_file=/scratch/tyoeasley/brain_representations/dist_mtx_list.csv

file_list=$(cat ${list_file})

flag=false
for fname in ${file_list}; do
	if [[ ${fname} == ${new_fname} ]]; then
		flag=true;
	fi;
done

if [[ ${flag} == false ]]; then
	echo ${new_fname} >> ${list_file}
fi
