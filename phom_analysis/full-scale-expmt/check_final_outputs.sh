#!/bin/bash

search_term=${1:-"phom_X.txt"}
empty_only=${2:-false}
subdir_term=${3:-"within_*"}

for i in $( ls ${subdir_term}/*_*/phom* -d)
do
	match_num=$(ls ${i}/${search_term} 2>/dev/null | wc -l)
	if $empty_only
	then
		if [ "$match_num" = "0" ]
		then
			echo "no match found in ${i}"
		fi
	else	
		echo "${i}: ${match_num} ${search_term} output files"
	fi
done
