graddir="/scratch/tyoeasley/brain_representations/gradient_reps"
phom_basedir="/scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within_grad"

for dim in {15,25,50,100,200,300}
do
	datadir="${graddir}/gradmaps_d${dim}"

	outfile="${phom_basedir}/grad${dim}_Maps/subj_list_Maps.csv"

	ls ${datadir}/emb_subj-??????_aligned.npy > $outfile
	# ls ${datadir}/emb_subj-??????.npy > $outfile
done

echo ""
echo "Checking number of entries in each subject list..."
for fname in $( ls grad*_Maps/subj_list_Maps.csv )
do 
	echo ""
	echo "${fname}:"
	cat $fname | wc -l 
done
