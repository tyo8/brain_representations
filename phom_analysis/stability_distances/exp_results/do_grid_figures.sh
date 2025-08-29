#/bin/bash

indir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull"
outdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull/figures"

srcdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/src_py/figures/permtest_dists"
nullpairs_srcpath="${srcdir}/summarize_nullpair_dists.py"
# nullpairs_srcpath="${srcdir}/summarize_nullpair_dists_stable.py"
extremals_srcpath="${srcdir}/extremal_nullpair_dists.py"
null_solo_srcpath="${srcdir}/single_null_dists.py"

# declare -a rstr=("Psim" "geodesic" "Maps" "Amps" "NM")	# restriction strings 
declare -a rstr=("^Psim" "Psim" "ICA*Psim" "Schaefer*Psim" "Maps*Psim" "NM*Psim" "Amps*Psim" "ICA" "Schaefer" "grad" "Amps" "NM" "Maps")	# restriction strings 
declare -a subsmp=("perm" "bstrap")			# sampling types
declare -a corrs=("fdr" "fwe")				# mulitple-comparision corrections
declare -a perms=("subject" "feature")			# permutation types
# declare -a alphas=(0.05 0.01)				# significance threshold values
declare -a alphas=(0.05 0.01 0.001)			# significance threshold values

solo_indir="$(dirname ${indir})"

python -X faulthandler ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "Psim" -c "fdr" -P "subject" -a 0.05 -L -X -w -v

for alpha in ${alphas[@]}
do
	python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}/ROC_analysis" -r "Psim" -a ${alpha} -w -v -R 
	python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}/ROC_analysis" -a ${alpha} -w -v -R 
	printf "\n#################################################################################################\n\n"
done

# python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "Psim" -c "fdr" -P "subject" -L -C -X -w -v
# python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "Psim" -c "fdr" -P "subject" -a 0.05 -L -X -w -v
# python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "Psim" -c "fdr" -P "subject" -a 0.05 -L -C -X -w -v
# python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "ICA*Psim" -c "fdr" -P "subject" -a 0.05 -L -C -X -w -v
printf "\n#################################################################################################\n\n"
# echo "debugging run complete. exiting."; exit


printf "## now looping through parameters arrays (in order): \n\${alphas} \n\${perms} \n\${corrs} \n\${rstr} \n\n"

for alpha in ${alphas[@]}
do
	for P in ${perms[@]}
	do
		for C in "${corrs[@]}"
		do
			# python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r "Psim" -c "${C}" -P "${P}" -a ${alpha} -L -V -X -w -v
			printf "\n#################################################################################################\n\n"
			# echo "debugging run complete. exiting."; exit

			python -X faulthandler ${nullpairs_srcpath} -i ${indir} -o ${outdir} -c "${C}" -P "${P}" -a ${alpha} -L -C -X -w -v
			printf "\n#################################################################################################\n\n"

			for R in "${rstr[@]}"
			do
				python -X faulthandler ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r ${R} -c ${C} -P ${P} -a ${alpha} -L -C -X -w -v
				printf "\n#################################################################################################\n\n"
				if [[ "${alpha}" == "0.001" ]]
				then
					python -X faulthandler ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r ${R} -c ${C} -P ${P} -L -C -X -w -v
					printf "\n#################################################################################################\n\n"
				fi
			done
		done
	done
done

printf "\n\n Grid figures computed. Exiting."
