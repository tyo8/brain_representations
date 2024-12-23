#!/bin/bash

search_term=${1:-"phom*/phom_X.txt"}
subdir_term=${2:-"within_*"}

for i in $( ls ${subdir_term}/*_*/permstrapping -d)
do 
	echo "${i}: $(ls ${i}/${search_term} 2>/dev/null | wc -l) ${search_term} output files"
done
