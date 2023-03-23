## Given a single filename, parses the file datatype and returns the numerical data stored within.
## Flattens all return data (including image/volumetric data) to a single dimension. 
import numpy as np
import re
import csv

def parse_fname(fname):
    components1 = re.split("([0-9]{6})",fname)
    fname_end = components1[-1]
    components2 = re.split(r"(\.)",fname_end)
    ext = ''.join(components2[1:])
    parser=switch(ext)
    data=parser(fname)
    return data.flatten() 

def switch(argument):
    switcher = {
        ".csv": load_csv,
        ".dscalar.nii": load_cifti,
        ".txt": load_txt
     }
    parser = switcher.get(argument, lambda argument: print("Unknown file extension: "+argument))
    return parser

def load_cifti(fname):
    import nibabel as nib
    data = nib.load(fname).get_fdata().flatten()
    return data

def load_csv(fname):
    data0 = np.genfromtxt(fname,delimiter=",")
    if len(data0.shape) == 2:
        if data0.shape[0] == data0.shape[1]:
            # assumes that square .csv matrices are symmetric and have uninformative diagonals
            data = upper_tri_indexing(data0) 
        else:
            data = data0.flatten()
    else:
        # does not modify .csv vectors
        data = data0 
    return data

def load_txt(fname):
    data0 = np.genfromtxt(fname)
    data = data0.flatten()
    return data

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]
