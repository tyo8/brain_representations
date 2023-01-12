#!/bin/sh

module load workbench
HCPdir=/HCP/OpenAccess/unzip/1200subject
MMP=/scratch/janine/HCP_MMP_1_Nov2017/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii

if [ ! -d MMP_netmats ]; then
	mkdir MMP_netmats
fi

for s in `cat HCP_IDs_rem.csv` ; do
   wb_command -cifti-parcellate ${HCPdir}/${s}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii ${MMP} COLUMN MMP_netmats/${s}_REST1_LR.ptseries.nii

   wb_command -cifti-parcellate ${HCPdir}/${s}/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii ${MMP} COLUMN MMP_netmats/${s}_REST1_RL.ptseries.nii
		
   wb_command -cifti-parcellate ${HCPdir}/${s}/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii ${MMP} COLUMN MMP_netmats/${s}_REST2_LR.ptseries.nii

   wb_command -cifti-parcellate ${HCPdir}/${s}/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii ${MMP} COLUMN MMP_netmats/${s}_REST2_RL.ptseries.nii
done


