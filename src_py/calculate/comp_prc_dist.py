import os
import re
import csv
import pickle
import numpy as np
from scipy.spatial import procrustes

list_path = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/BR_label_list.csv"

def main(argvals):
    PRCd.iter_PRCd_list(argvals[1],argvals[2],listpath=argvals[3])

def iter_PRCd_list(dataset_list_name,output_dir,listpath=list_path):
    with open(dataset_list_name,newline='') as fin:
        dataset_lists = list(csv.reader(fin))

    dataset_list = list(map(''.join,dataset_lists))
    n_reps = len(dataset_list)
    
    with open(listpath,newline='') as fin:
        listpath_lists = list(csv.reader(fin))

    namelist = list(map(''.join,listpath_lists))
    
    for i in range(n_reps):
        for j in range(i+1,n_reps):
            xname = dataset_list[i]
            yname = dataset_list[j]
            X = np.loadtxt(xname, delimiter=",")
            Y = np.loadtxt(yname, delimiter=",")
            
            results = comp_PRCd(X,Y)

            save_results(results,output_dir,namelist[i],namelist[j])

# given a pair X and Y of NxN distance matrices, computes the Procrustes distance between them.
def comp_PRCd(X,Y):
    # "X_std"       standardized (centered & whitened) version of X 
    # "Y_Xstd"      re-oriented Y that is maximally aligned with X_std
    # "disparity"   orientation disparity: Frobenius norm of difference matrix D=X_std-Y_Xstd
    X_std, Y_Xstd, disparity = procrustes(X,Y)   

    results = [X_std, Y_Xstd, disparity]
    return results 


def save_results(results,output_dir,xname,yname):
    fname = xname + "_and_" + yname + ".prc"
    fpath = output_dir + "/" + fname

    with open(fpath,'wb') as results_fpath:
        pickle.dump(results,results_fpath,protocol = 4)

if __name__="__main__":
    main(sys.argv)
