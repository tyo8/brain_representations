#!/bin/sh

searchdir=${1: }
varnames=$(find ${searchdir} *.CCA_res)

source /export/anaconda/anaconda3/anaconda3-2020.07/bin/activate Stats

for i in $varnames; do
	echo $i >> regularizations.csv
	python pull_reg_val.py $i >> regularizations.csv
done	
