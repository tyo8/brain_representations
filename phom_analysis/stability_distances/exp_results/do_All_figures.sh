#/bin/bash

indir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull"
outdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull/figures"

srcdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/src_py/figures/permtest_dists"
nullpairs_srcpath="${srcdir}/summarize_nullpair_dists.py"
extremals_srcpath="${srcdir}/extremal_nullpair_dists.py"
null_solo_srcpath="${srcdir}/single_null_dists.py"

declare -a rstr=("Maps" "Psim" "geodesic" "Amps" "NM")
declare -a subsmp=("perm" "bstrap")
declare -a corrs=("fdr" "fwe")
declare -a perms=("subject" "feature")

solo_indir="$(dirname ${indir})"

# 'solo' figures (-S flag): full null+bootstrap distance distributions for every brain representation
python ${null_solo_srcpath} -v -w -i ${solo_indir} -o "${solo_indir}/figs_null" -D -L

# 'solo pair' figures (-S flag): full null+bootstrap paired-distance distributions for every representation pair
python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -S -C -D

python ${null_solo_srcpath} -v -w -i ${solo_indir} -o "${solo_indir}/figs_null" -A -S -L
python ${extremals_srcpath} -v -w -i ${indir} -o ${outdir} -L
python ${extremals_srcpath} -v -w -i ${indir} -o ${outdir} -L -E

printf "#################################################################################################\n\n"

for C in ${corrs[@]}
do
	for P in ${perms[@]}
	do
		python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -c "${C}" -P "${P}" -C
		printf "#################################################################################################\n\n"
	done
done

printf "## now looping through arrays: \n${rstr} \n${corrs} \n\n"
for R in "${rstr[@]}"
do
	python ${extremals_srcpath} -v -w -r ${R} -i ${indir} -o ${outdir} -L
	printf "#################################################################################################\n\n"
	python ${extremals_srcpath} -v -w -r ${R} -i ${indir} -o ${outdir} -L -E
	printf "#################################################################################################\n\n"
   for C in "${corrs[@]}"
   do
	   python ${nullpairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -c ${C} -r ${R} -P "subject" -D -C
	   printf "#################################################################################################\n\n"
   done
   for T in "${subsmp[@]}"
   do
	   python ${null_solo_srcpath} -v -w -i ${solo_indir} -o "${solo_indir}/figs_null" -r ${R} -t ${T} -A -L
	   printf "#################################################################################################\n\n"
   done
done
