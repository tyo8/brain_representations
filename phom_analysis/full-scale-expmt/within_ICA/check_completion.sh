#!/bin/sh

out_type=${1:-"dist"}

for dname in $( ls ICA* -d )
do 
	echo $dname
	ls $dname/phom_data*/${out_type}*/* | wc -l 
done
