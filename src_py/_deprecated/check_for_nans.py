import os
import sys
import glob
import numpy as np
import termplotlib as tpl


def check_datadir(datadir):
    flist = glob.glob(os.path.join(datadir,'sub-*.csv'))
    if not flist:
        flist = glob.glob(os.path.join(datadir,'subj-*.csv'))
        if not flist:
            flist = glob.glob(os.path.join(datadir,'sub-*.txt'))
            if not flist:
                flist = glob.glob(os.path.join(datadir,'subj-*.txt'))
                if not flist:
                    flist = glob.glob(os.path.join(datadir,'*series.nii'))

    num_subj = len(flist)
    
    nans_per_subj = np.asarray([np.count_nonzero(np.isnan(_load_data(fpath))) for fpath in flist])
    print("number of subjects with nans in data:", np.count_nonzero(nans_per_subj))
    print("distribution of numbers of nans:")
    counts, bin_edges = np.histogram(nans_per_subj)
    fig=tpl.figure()
    fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
    fig.show()


def _load_data(fpath):
    if "series.nii" in fpath:
        import nibabel as nib
        data = nib.load(fpath).get_fdata()
    else:
        data = np.loadtxt(fpath)

    return data

if __name__=="__main__":
    dname_in = sys.argv[1]
    check_datadir(dname_in)
