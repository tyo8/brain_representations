#!/bin/sh

subj_nums_list=$(ls sub*run-*.csv | cut -d- -f2 | cut -d_ -f1 | sort -u)

for subj_num in ${subj_nums_list}
do
	fname=Amps_cat/subj-${subj_num}_catrun.csv
	cat sub-${subj_num}_run-*.csv >> ${fname}
done
