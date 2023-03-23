ICAdir="/scratch/tyoeasley/brain_representations/ICA_reps"
phom_basedir="/scratch/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within_ICA"

for feat in {"Amplitudes","partial_NMs"}
do
	if [[ "${feat}" == "Amplitudes" ]]
	then
		subfeat="Amps"
	fi
	if [[ "${feat}" == "partial_NMs" ]]
	then
		subfeat="pNMs"
	fi
	for dim in {15,25,50,100,200,300}
	do
		datadir="${ICAdir}/3T_HCP1200_MSMAll_d${dim}_ts2_Z/${feat}"

		outfile="${phom_basedir}/ICA${dim}_${subfeat}/subj-list-${subfeat}.csv"

		ls ${datadir}/sub-*.csv > $outfile
	done
done
