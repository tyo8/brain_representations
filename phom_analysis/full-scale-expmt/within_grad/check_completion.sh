#!/bin/sh

out_type=${1:-"dist"}

for dname in $( ls grad* -d )
do 
	echo "${dname} ${out_type}:"
	ls $dname/phom_data*/${out_type}*/* | wc -l 
done
