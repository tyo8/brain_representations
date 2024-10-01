import os
import csv
import shutil
import numpy as np


########################################################################################################################################
# input: path to .csv file listing representation-wise filenames of .csv files listing paths to subject data files
# output: sll ubject data for all brain representations in list
def load_reps(dataset_listname):
    print(dataset_listname)
    with open(dataset_listname,newline='') as fin:
        dataset_lists = list(csv.reader(fin))

    dataset_list = list(map(''.join,dataset_lists))
    n_reps = len(dataset_list)

    reps = [None]*n_reps

    for i in range(n_reps):
        reps[i] = _parse_dataset(dataset_list[i])
        print(eps[i].shape)

    return reps


# loads brain representations data from file of list of filenames and reduces data dimensionality via SVD; written for use with lindecomp module
def load_reduced_reps(dataset_listname):
    reps = load_reps(dataset_listname)
    n_reps = len(reps)

    Rreps = [None]*n_reps

    for i in range(n_reps):
       R,V = svd_reduce(reps[i])
       Rreps[i] = R

    return Rreps



# input: path to .csv file listing paths to subject data files for a single representation
# ouput: all subject data for a single brain representation
def _parse_dataset(dataset_name):
    ## debug code:
    # print("path to dataset: "+dataset_name)

    datadir = os.path.dirname(dataset_name)
    with open(dataset_name,newline='') as fin:
        dataname_lists = list(csv.reader(fin))

    dataname_list = list(map(''.join,dataname_lists))
    data = _parse_datalist(dataname_list,datadir)

    return data


# reads in a list of N datafile names and returns an NxD matrix (D = data dimension)
def _parse_datalist(dataname_list,datadir = "."):
    n_subj = len(dataname_list)
    n_feat = len(_parse_fname(os.path.join(datadir,dataname_list[0])))

    data = np.zeros((n_subj,n_feat))

    for i in range(n_subj):
        dataname = os.path.join(datadir,dataname_list[i])
        data[i,:] = _parse_fname(dataname)

    return data

def _parse_fname(fname):
    ext = os.path.splitext(os.path.basename(fname))[1]
    _parser = _switch(ext)
    data = _parser(fname)
    data = _shaper(data)
    return np.single(data.flatten())

def _switch(argument):
    _switcher = {
        ".csv": _load_csv,
        ".txt": _load_txt,
        ".npy": _load_npy,
        ".nii": _load_scan,
        ".gz": _load_scan
     }
    _parser = _switcher.get(argument, lambda argument: print("Unknown file extension: "+argument))
    return _parser

def _load_scan(fname):
    import nibabel as nib
    sdata = nib.load(fname)
    data = np.asarray(sdata.get_fdata()).flatten()
    return data

def _load_csv(fname):
    try:
        data = np.loadtxt(fname)
    except ValueError:
        data = np.loadtxt(fname, delimiter=",")
    return data

def _load_txt(fname):
    try:
        data = np.loadtxt(fname)
    except ValueError:
        data = np.loadtxt(fname, delimiter=",")
    return data

def _load_npy(fname):
    data = np.load(fname, allow_pickle=True)
    return data

def _shaper(data):
    if len(data.shape) == 2:
        if data.shape[0]==data.shape[1]:
            if np.allclose(data, data.T):
                data = triu_vals(data)       # assumes that symmetric matrices have noninformative diagonals
            else:
                data=data.flatten()
        else:
            data = data.flatten()
    return data

def triu_vals(A):
    n = A.shape[0]
    vals = A[np.triu_indices(n,1)]
    return vals

# for an NxP matrix X with P > N, returns a pair R and V of NxN and PxN matrices such that X = RV' and V unitary.
# if P <= N, returns R = X and V = PxP identity.
def svd_reduce(X):
    n_subj = X.shape[0]
    n_dim = X.shape[1]

    if n_dim <= n_subj:
        R = X
        V = np.identity(n_dim)
    else:
        U,s,Vh = np.linalg.svd(X,full_matrices = False)
        R = U @ np.diag(s)
        V = np.matrix.getH(Vh)
        ## debugging code:
        # print("Using SVD reduction. Maximum singular value = " + str(s[0]))

    return R,V
########################################################################################################################################


########################################################################################################################################
# saves CIFTI2 version of diffusion embedding representation
def export_cifti(outpath, data, headers_list):
    # change file extension to .dtseries.nii
    ext = os.path.splitext(os.path.basename(outpath))[1]
    outpath = outpath.replace(ext, '.dtseries.nii')

    header = headers_list[0]
    nifti_header = headers_list[1]

    # assumes len(data.shape)==2 and data.shape[1] is unmodified from read-in cifti data
    ts_axis, brain_axis = [header.get_axis(i) for i in range(len(data.shape))]

    # change series axes of template data to match data
    new_ts_axis = nib.cifti2.cifti2_axes.SeriesAxis(0, 1, data.shape[0])
    # update cifti header
    new_header = (new_ts_axis, brain_axis)

    # create cifti template data
    new_cdata = nib.Cifti2Image(data, header=new_header, nifti_header=nifti_header)
    # export data to cifti format
    new_cdata.to_filename(outpath)
########################################################################################################################################


########################################################################################################################################
# checks for the existence of multiple directories: makes them if they do not exist
def check_to_make_dirs(dirlist):
    for dirloc in dirlist:
        if not os.path.isdir(dirloc):
            os.makedirs(dirloc)
########################################################################################################################################


########################################################################################################################################
# loads list of human-readable labels for datasets
def load_namelist(listpath):
    with open(listpath,newline='') as fin:
        listpath_lists = list(csv.reader(fin))

    namelist = list(map(''.join,listpath_lists))

    return namelist
########################################################################################################################################


########################################################################################################################################
# computes partial network matrices of a dataset
def comp_partial_netmats(data):
    C = np.cov(data)
    pcorr = np.linalg.pinv(C, hermitian=True)
    return pcorr
########################################################################################################################################


########################################################################################################################################
# given a similarity matrix "sim_mtx", computes a p-distance matrix "dist_mtx" (typically, p=1 or p=2)
def p_dist(sim_mtx, order=1, noise=1e-12):
    tmp = np.power(np.abs(sim_mtx - noise), order, dtype=float)
    dist_mtx = np.power(1 - tmp, 1/order)
    np.fill_diagonal(dist_mtx, 0)
    return dist_mtx
########################################################################################################################################
