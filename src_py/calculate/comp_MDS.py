import csv
import sys
import pickle
import numpy as np
import sklearn.manifold as mnf
import SubjSubjHomologies as SSH

list_path = "/scratch/tyoeasley/brain_representations/BR_label_list.csv"

def main(argvals):
    listiter_MDS(argvals[1], argvals[2], listpath = argvals[3])

def listiter_MDS(dist_list_name,output_dir,listpath = list_path,normalize = True,n_comps = [2,3]):
    with open(dist_list_name,newline='') as fin:
        dist_lists = list(csv.reader(fin))

    with open(listpath,newline='') as fin:
        listpath_lists = list(csv.reader(fin))

    dist_list = list(map(''.join,dist_lists))
    dist_mtx_array = SSH.load_dist_mtxs(dist_list,normalize)

    namelist = list(map(''.join,listpath_lists))

    # n_comps = int(np.logspace(1,2.8,10))
    n_comps = [2,3]

    for j in n_comps:
        for i in range(dist_mtx_array.shape[0]):
            X = dist_mtx_array[i,:,:]
            results,n_comp = comp_MDS(X,n_comps = j)

            save_results(results,output_dir,namelist[i],n_comp)


# Given an NxN distance matrix X, computes the mutlidimensional scaling of X (an immersion of its datapoints)
def comp_MDS(X,n_comps = 0):
    if n_comps == 0:
        n_comps = int(np.sqrt(X.shape[0]))
    else:
        n_comps = int(n_comps)

    MDS_model = mnf.MDS(n_components = n_comps, dissimilarity = 'precomputed')
    X_embedded = MDS_model.fit_transform(X)

    results = [X_embedded, MDS_model]
    return results, n_comps


def save_results(results,output_dir,xname,n_comps):
    ext   = ".mds_"
    fname = xname + ext + "c" + str(int(n_comps))
    fpath = output_dir + "/" + fname

    with open(fpath,'wb') as results_fpath:
        pickle.dump(results,results_fpath,protocol = 4)


if __name__=="__main__":
    main(sys.argv)
