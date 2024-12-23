import os
import sys
import csv
import nilearn
import numpy as np
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker as NLM

def main(argvals):
    n_roi=int(argvals[1])
    
    subj_list_fname=argvals[2]
    with open(subj_list_fname, newline='') as fin:
        subj_list_list=list(csv.reader(fin))

    subj_list=list(map(''.join,subj_list_list))

    out_dir=argvals[3]

    if len(argvals) > 4:
        HCPdir=argvals[4]

    ext_schaefer_dts(n_roi, subj_list, out_dir, HCPdir=HCPdir)


def ext_schaefer_dts(n_roi, subj_list, out_dir,
        HCPdir='/scratch/tyoeasley/profumo_anls/DRmaps_profumo_subjs'):

    sch_dataset = datasets.fetch_atlas_schaefer_2018(n_rois=n_roi, yeo_networks=17)
    atlas_filename = sch_dataset.maps
    labels = sch_dataset.labels

    masker = NLM(labels_img=atlas_filename, standardize=True, memory='nilearn_cache', verbose=5)

    ## pre-PROFUMO convention
    fileloc = [os.path.join(HCPdir, 'SCA_sub-' + i + '.nii.gz') for i in subj_list]

    ## HCP CIFTI convention
    #fileloc = [os.path.join(HCPdir, i,
    #    'MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii')
    #    for i in subj_list]
    
    outloc = [os.path.join(out_dir, 'sub-' + i + '.dtseries.nii') for i in subj_list]

    for i in range(len(fileloc)):
        timeseries = masker.fit_transform(fileloc[i])
        np.savetxt(outloc[i], timeseries)


if __name__=="__main__":
    main(sys.argv)
