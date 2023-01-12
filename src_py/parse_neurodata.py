## Given a single filename, parses the file datatype and returns the numerical data stored within.
## Flattens all return data (including image/volumetric data) to a single dimension. 


import nibabel as nib
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
        ".dscalar.nii": load_cifti_profumo_map,
        ".txt": load_txt
     }
    parser = switcher.get(argument, lambda argument: print("Unknown file extension: "+argument))
    return parser

def load_cifti_profumo_map(fname):
    map_mat = nib.load(fname)
    mtx = np.array(map_mat.dataobj.get_unscaled())
    data = mtx[6,:]
    # data = mtx.flatten()
    return data

def load_csv(fname):
    data0 = np.genfromtxt(fname,delimiter=",")
    if len(data0.shape) == 2:
        if data0.shape[0] == data0.shape[1]:
            data = upper_tri_indexing(data0) # assumes that square .csv matrices are symmetric and have uninformative diagonals
        else:
            data = data0.flatten()
    else:
        data = data0 # does not modify .csv vectors
    return data

def load_txt(fname):
    data0 = np.genfromtxt(fname)
    data = data0.flatten()
    return data

def upper_tri_indexing(A):
    m = A.shape[0]
    r,c = np.triu_indices(m,1)
    return A[r,c]
