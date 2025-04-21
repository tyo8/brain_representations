#/bin/bash

indir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull"
outdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull/figures"

srcdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/src_py/figures/permtest_dists"
nullpairs_srcpath="${srcdir}/summarize_nullpair_dists.py"
extremals_srcpath="${srcdir}/extremal_nullpair_dists.py"
null_solo_srcpath="${srcdir}/single_null_dists.py"

declare -a rstr=("Psim" "geodesic" "Maps" "Amps" "NM" "inner")	# restriction strings 
declare -a subsmp=("perm" "bstrap")				# sampling types
declare -a corrs=("fdr" "fwe")					# mulitple-comparision corrections
declare -a perms=("subject" "feature")				# permutation types
declare -a alphas=(0.01 0.05)					# significance threshold values

solo_indir="$(dirname ${indir})"

for P in ${perms[@]}
do
	for C in ${corrs[@]}
	do
		for alpha in ${alphas[@]}
		do
			python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -c "${C}" -P "${P}" -a ${alpha} -L -C -w -v
			printf "\n#################################################################################################\n\n"
		done
	done
	python ${extremals_srcpath} -i ${indir} -o ${outdir} -P ${P} -L -v -w
	python ${extremals_srcpath} -i ${indir} -o ${outdir} -P ${P} -L -E -v -w
	printf "\n#################################################################################################\n\n"
done

# 'solo' figures (-S flag): full null+bootstrap distance distributions for every brain representation
python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}/figs_null" -D -A -L -v -w
printf "\n#################################################################################################\n\n"
python ${extremals_srcpath} -i ${indir} -o ${outdir} -P "subject" -L -E -v -w

for alpha in ${alphas[@]}
do
	python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}" -a ${alpha} -v -w -R 
	printf "\n#################################################################################################\n\n"
done

for R in "${rstr[@]}"
do
	python ${null_solo_srcpath} -v -w -i ${solo_indir} -o "${solo_indir}/figs_null" -r ${R} -L -A -D
	printf "\n#################################################################################################\n\n"
done

printf "## now looping through arrays: \n${rstr} \n${corrs} \n\n"
for R in "${rstr[@]}"
do
	for P in ${perms[@]}
	do
		for C in "${corrs[@]}"
		do
			for alpha in ${alphas[@]}
			do
				python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r ${R} -c ${C} -P ${P} -a ${alpha} -L -C -w -v
				printf "\n#################################################################################################\n\n"
			done
			python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -r ${R} -c ${C} -P ${P} -L -C -w -v
			printf "\n#################################################################################################\n\n"
		done
		python ${extremals_srcpath} -i ${indir} -o ${outdir} -r ${R} -P ${P} -L -E -v -w
		printf "\n#################################################################################################\n\n"
	done
	for T in "${subsmp[@]}"
	do
		python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}/figs_null" -r ${R} -t ${T} -A -L -D -v -w
		printf "\n#################################################################################################\n\n"
	done
done

# 'solo pair' figures (-S flag): full null+bootstrap paired-distance distributions for every representation pair
python ${null_solo_srcpath} -i ${solo_indir} -o "${solo_indir}/figs_null" -Z -S -v -w
python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -L -S -w -v
printf "\n#################################################################################################\n\n"
