import os
import sys
import nibabel as nb
import numpy as np


def main(parent_dir, ext='.ptseries.nii'):
    flist = [i for i in os.listdir(parent_dir) if ext in i]
    for fname in flist:
        fpath = os.path.join(parent_dir,fname)
        parcellated_cifti = nb.load(fpath)
        parcel_data = np.asarray(parcellated_cifti.get_fdata())

        out_path = fpath.replace(ext, '.csv')
        np.savetxt(out_path, parcel_data, fmt='%.4f')


if __name__=="__main__":
    parent_dir = sys.argv[1]

    ext='.ptseries.nii'
    if len(sys.argv) > 2:
        ext = sys.argv[2]

    main(parent_dir, ext=ext)
