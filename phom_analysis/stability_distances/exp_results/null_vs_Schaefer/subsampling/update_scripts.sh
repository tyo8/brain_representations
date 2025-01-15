#!/bin/bash

submit=${1:-false}
script_name=${2:-"do_bsdists"}
dir_pattern_type=${3:-"*_vs_*"}
base=${4:-$(pwd)}

echo ""
echo "Updating scripts matching \"${base}/${dir_pattern_type}/${script_name}\""
echo "Updating from generic version at: \"$(ls ${base}/${script_name})\""
echo "Submitting scripts after update: ${submit}"
echo ""

for i in $(ls ${base}/${dir_pattern_type} -d)
do
	Xname=$( basename ${i} | cut -d_ -f 1 )
	Yname=$( basename ${i} | cut -d_ -f 3 )

	cp ${base}/${script_name} ${i}/${script_name}

	sed -i "s/{Xname}/${Xname}/g" ${i}/${script_name}
	sed -i "s/{Yname}/${Yname}/g" ${i}/${script_name}

	if [ ${Yname} = "self" ]
	then
		sed -i "/within_self/d" ${i}/${script_name}
		sed -i "s/Ydistlist/Xdistlist/g" ${i}/${script_name}
	fi

	echo "Name of toy model: ${Xname}_vs_${Yname} (updated!)"
	if $submit
	then
		cd ${i}
		echo "Submitting: $(ls ${i}/${script_name})"
		sbatch ${i}/${script_name}
		echo ""
	fi
done

echo ""
