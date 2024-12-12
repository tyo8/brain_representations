#!/bin/sh

out_type=${1:-"dist"}

for dname in $( ls Yeo* -d )
do 
	echo "In $dname over:"
	for i in $(ls $dname/phom_data*/${out_type}* -d)
	do
		echo ${i}
		ls ${i} | wc -l 
	done
	echo ""
done
