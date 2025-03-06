#/bin/bash

indir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull"
outdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/All_vs_AllNull/figures"

srcdir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/src_py/figures/permtest_dists"
pairs_srcpath="${srcdir}/DEV_exp_results_pairs.py"
nullpairs_srcpath="${srcdir}/DEV_extremal_null_dists.py"
null_solo_srcpath="${srcdir}/DEV_exp_results_null.py"

declare -a rstr=("Maps" "Psim" "geodesic" "NMs")
declare -a corrs=("fdr" "fwe")
declare -a subsmp=("perm" "bstrap")

solo_indir="$(dirname ${indir})"

python ${nullpairs_srcpath} -v -w -i ${indir} -o ${outdir} -L
python ${nullpairs_srcpath} -v -w -i ${indir} -o ${outdir} -L -E
python ${pairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -c "fwe"
python ${pairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -c "fdr"

printf "## now looping through arrays: \n${rstr} \n${corrs} \n\n"
for R in "${rstr[@]}"
do
	printf "#################################################################################################\n\n"
	python ${nullpairs_srcpath} -v -w -r ${R} -i ${indir} -o ${outdir} -L
	printf "#################################################################################################\n\n"
	python ${nullpairs_srcpath} -v -w -r ${R} -i ${indir} -o ${outdir} -L -E
   for C in "${corrs[@]}"
   do
	   printf "#################################################################################################\n\n"
	   python ${pairs_srcpath} -i ${indir} -o ${outdir} -w -v -L -c ${C} -r ${R}; 
   done
   for T in "${subsmp[@]}"
   do
	   printf "#################################################################################################\n\n"
	   python ${null_solo_srcpath} -v -w -i ${solo_indir} -o "${solo_indir}/figs_null" -r ${R} -t ${T}
   done
done
