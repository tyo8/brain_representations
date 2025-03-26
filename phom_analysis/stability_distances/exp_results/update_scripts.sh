#!/bin/bash

submit=${1:-false}
permtype=${2:-"feat"}
script_name=${3:-"do_permdists_null"}
dir_pattern_type=${4:-"within_*/permtesting"}
base=${5:-$(pwd)}

echo ""
echo "Using permtutation type: ${permtype}"
echo "Updating scripts matching \"${base}/${dir_pattern_type}/${script_name}\""
echo "Updating from generic version at: \"$(ls ${base}/${script_name})\""
echo "Submitting scripts after update: ${submit}"
echo ""

for i in $(ls ${base}/${dir_pattern_type} -d)
do
	Xname=$( basename $( dirname ${i} ) | cut -d_ -f 2)

	cp ${base}/${script_name} ${i}/${script_name}
	echo "updated ${i}/${script_name}, permtest distances for modality ${Xname}"

	sed -i "s/{Xname}/${Xname}/g" ${i}/${script_name}
	sed -i "s/{permtype}/${permtype}/g" ${i}/${script_name}

	if $submit
	then
		cd ${i}
		echo "submitting ${script_name} in $(pwd)"
		sbatch ${script_name}
		echo ""
	fi
done

cd ${base}
