import os
import csv
import sys
import time
import dill
import shutil
import inspect
import datetime
import numpy as np
import lindecomp_brainrep as ld_br

## assigns named variables to user inputs and runs primary task
def main(argvals):
    # filepath to list of names of datasets
    dataset_fname=argvals[1]
    if ".csv" in dataset_fname:
        reps = HCP_utils.load_reduced_data(dataset_fname)
    elif ".brep" in dataset_fname:
        with open(dataset_fname,'rb') as fin:
            reps = dill.load(fin)
    else:
        sys.exit("Unrecognized file extension for brain representation data.")

    # base directory in output tree
    output_basedir=argvals[2]

    # path to saved, precomputed list of (allowed) permuations
    permpath=argvals[3]
    # read in set of heteroscedastic permutations; convert to int type for indexing; subtract 1 for index differences with matlab
    permset = np.genfromtxt(permpath,delimiter=",").astype(int) - 1


    # path to list of brain representation names (same order as datasets are listed in dataset_fname)
    listpath=argvals[4]
    namelist=HCP_utils.load_namelist(listpath)

    # path to list of paired [pairname, optimal_regularization] values
    regpath=argvals[5]
    # read in paired [pair names, regularization hyperparameters] list
    with open(regpath,newline='') as fin:
        reglist = list(csv.reader(fin))
        ## debug code
        print(reglist)
        # testval = ld_br.find_reg_val(reglist,namelist[0],namelist[1])
        # print(testval)


    # linear decomposition algorithm: currently either CCA or PLS (far more likely to be CCA)
    decomp_method=argvals[6]

    # number of threads requested by job
    if len(argvals) > 7:
        n_workers = int(argvals[7])
    else:
        n_workers = 1

    pt_arch_dir = run_permutation_testing(reps, output_basedir, permset, namelist, reglist, decomp_method, n_workers)
    return pt_arch_dir


## Distributes, runs, and saves the results of linear decompositions on permuted data.
##
## INPUTS:
## --reps:              brain representations (datasaets) on which to pairwise perform CCA; must be nonempty if "dataset_list_name" is empty
## --output_basedir:    base directory storing results of permutation testing 
## --permset:           NxP array of indices, where N is the number of subjects and P the number of permutations
## --namelist:          list of dataset (equiv: brain representation) names
## --reglist:           paired list of regularization hyperparameter values for each variable pair
## --decomp_method:     linear decomposition algorithm, specified as a string; default 'CCA', but 'PLS' is also (nominally) supported
## --n_workers:         number of parallel workers (specified at job submission)
def run_permutation_testing(reps, output_basedir, permset, namelist, reglist, decomp_method, n_workers):
    ## implements per-worker permutation testing
    def par_pt_iters(worker_spec):
        worker_output_basedir = worker_spec[0]
        worker_permset = worker_spec[1]

        permutation_testing(worker_output_basedir, reps, worker_permset, reglist,
                namelist, decomp_method=decomp_method, chkpt=True)


    ## initializes parallel-sliced inputs and collects outputs from parallel permutation tests
    def run_parallel_permtests(output_basedir, n_workers):
        import pathos.multiprocessing as mp

        # names a set of intermediate directories (by worker and date) to store parallel outputs before aggregation
        date_label = datetime.date.today().strftime("%Y%b%d")
        dirlocs = [os.path.join(output_basedir,decomp_method+"_pt_" + str(k) +"_" + date_label) for k in range(n_workers)]

        # splits permutation set over number of workers
        n_perms = permset.shape[1]
        permset_split = np.array_split(permset,n_workers,axis=1)

        # formats [intermediate directory, permutation set split] pairs to map into per-worker permutation testing function
        worker_spec = [[dirlocs[i], permset_split[i]] for i in range(n_workers)]

        tic = time.time()
        # initializes parallel pool and maps function & data to workers
        with mp.ProcessingPool(n_workers) as P:
            P.map(par_pt_iters,worker_spec)

        toc = time.time() - tic
        print("Elapsed time over " + str(n_workers) + " workers and " + str(n_perms) + " total permutations is " + str(toc) + " seconds.")

        # aggregates data from across intermediate worker storage directories
        agg_data,fpairnames = aggregate_cancorrs(dirlocs)
        pairnames = [i.split("_cancorr_nulldist")[0] for i in fpairnames]

        # overarching directory storing data aggregated over worker outputs
        pt_arch_dir = os.path.join(output_basedir,decomp_method+"_permtests_" + date_label)
        if not os.path.isdir(pt_arch_dir):
            os.makedirs(pt_arch_dir)
            print("WARNING: Created directory " + pt_arch_dir)

        # saves aggregated data in overarching directory
        for i in range(len(pairnames)):
            save_permtests(pt_arch_dir,pairnames[i],agg_data[i])

        return pt_arch_dir

    # run permutation tests in parallel -- calls all functions defined in this module using semiglobal variables
    pt_arch_dir = run_parallel_permtests(output_basedir,n_workers)
    print("Final results saved to directory " + pt_arch_dir)
    return pt_arch_dir


## Runs and saves the results of linear decompositions on permuted data.
def permutation_testing(output_basedir, reps, permset, reglist, namelist, 
        decomp_method='CCA', chkpt=True):
    # number of permutations
    n_perms = permset.shape[1]

    # number of datasets (equiv: brain_representations)
    n_reps = len(reps)

    # assigns correct linear decomposition function to name "lindecomp" (either 'CCA' or 'PLS')
    lindecomp = ld_br.switch(decomp_method)

    for i in range(n_reps):
        for j in range(i+1,n_reps):
            # computes pairname from pair of names
            pairname = namelist[i] + "_and_" + namelist[j]
            
            # selects data pair
            X = reps[i]
            Y_base = reps[j]

            # pulls pairname-associated regularization hyperparameter from paired reg_val list
            reg_val = ld_br.find_reg_val(reglist,namelist[i],namelist[j])

            # initializes null distribution of canonical correlations
            cancorr_nulldist_ij = [None]*n_perms

            for k in range(n_perms):
                # selects k-th permutation and computes associated permutation of Y data
                perm_idx = permset[:,k]
                Y = Y_base[perm_idx,:]
                
                # performs CCA with permuted data and extracts canonical correlation values
                CCA_res = lindecomp(X,Y,reg_val=reg_val,param_search=False)
                cancorrs = CCA_res.cancorrs

                # collects k-th vector of canonical correlations
                cancorr_nulldist_ij[k] = cancorrs

                # saves checkpoint every 500 iterations
                if np.mod(k+1,500) == 0:
                    if chkpt:
                        save_permtests(output_basedir,pairname,cancorr_nulldist_ij[:k])
            
            # save distributions to output directory for given data pair
            save_permtests(output_basedir,pairname,cancorr_nulldist_ij)

## saves null distribution canonical correlations to output directory for a given data pair
def save_permtests(output_basedir,pairname,cancorr_nulldist):
    if not os.path.isdir(output_basedir):
        os.makedirs(output_basedir)
        print("WARNING: directory " + output_basedir + " not found; new directory created.")
    
    # computes filename and filepath from base output directory and data pair name
    fname = pairname + "_cancorr_nulldist.csv"
    floc = os.path.join(output_basedir,fname)

    # writes list of canonical correlation vectors to .csv
    with open(floc,'w') as fout:
        write = csv.writer(fout)
        write.writerows(cancorr_nulldist)

## aggregates worker data and clears per-worker directories
def aggregate_cancorrs(dirlocs, del_dir = True):
    # locate null distribution filenames in target directory
    fpairnames = [fn for fn in os.listdir(dirlocs[0]) if "_cancorr_nulldist.csv" in fn]

    # number of (unique) data pairs
    n_pairs = len(fpairnames)

    # initialize aggregated data
    agg_data = [None]*n_pairs

    ## debug code:
    # print(fpairnames)

    for i in range(n_pairs):
        # initialize empty list to join to
        cancorrs = []
        for j in dirlocs:
            # compute filepath to data and read in data
            dataloc = os.path.join(j,fpairnames[i])
            with open(dataloc,'r') as fin:
                data = list(csv.reader(fin))

            # join loaded data to list of canonical correlations
            cancorrs += data

        # collect canonical correlations in array (and ensure that values are numerical)
        cancorrs_array = np.asarray([np.genfromtxt(i) for i in cancorrs])

        # collect i-th array of canonical correlations
        agg_data[i] = cancorrs_array

    # check 'del_dir' flag
    if del_dir:
        for i in dirlocs:
            # remove directory from which worker data has been aggregated
            shutil.rmtree(i)

    # return aggregated data and filenames for data pairs
    return agg_data,fpairnames

## checks if this function has been called by standard in
if __name__=="__main__":
    main(sys.argv)
