#!/bin/bash

submit=${1:-false}
script_name=${2:-"do_bash_phom_bootstraps"}
dir_pattern_type=${3:-"within_*"}
base=${4:-$(pwd)}

echo ""
echo "Updating scripts matching \"${base}/${dir_pattern_type}/${script_name}\""
echo "Updating from generic version at: \"$(ls ${base}/${script_name})\""
echo "Submitting scripts after update: ${submit}"
echo ""

for i in $(ls ${base}/${dir_pattern_type} -d)
do
	name=$( basename ${i} | cut -d_ -f 2 )
	if [ "${name}" = "All" ] || [ "${name}" = "{name}" ] # && ! [ "${script_name}" = "do_match_collate" ]
	then
		continue
	fi
	cp ${base}/${script_name} ${i}/${script_name}
	sed -i "s/{name}/${name}/g" ${i}/${script_name}
	echo "Name of toy model: ${name} (updated!)"
	if [[ "${name}" == *"u0" ]]  &&   [ "${script_name}" = "do_distmtx_comps" ]
	then
		sed -i "s/\-S/\-S \-O/g" ${i}/${script_name}
		echo "(added origin flag)"
	fi
	if $submit
	then
		cd ${i}
		echo "Submitting: $(ls ${i}/${script_name})"
		sbatch ${i}/${script_name}
		echo ""
	fi
done

echo ""
