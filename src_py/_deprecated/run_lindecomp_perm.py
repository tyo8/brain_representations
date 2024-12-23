import os
import sys
import csv
import time
import dill
import datetime
import numpy as np
import HCP_utils
import lindecomp_permtest as ld_pt


def main(argvals):
    ## assigns named variables to user inputs and runs primary task

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
        # testval = ld_pt.ld_br.find_reg_val(reglist,namelist[0],namelist[1])
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



def run_permutation_testing(reps, output_basedir, permset, namelist, reglist, decomp_method, n_workers):        
    ## implements per-worker permutation testing
    def par_pt_iters(worker_spec):
        worker_output_basedir = worker_spec[0]
        worker_permset = worker_spec[1]

        ld_pt.permutation_testing(worker_output_basedir, reps, worker_permset, reglist,
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
        agg_data,fpairnames = ld_pt.aggregate_cancorrs(dirlocs)
        pairnames = [i.split("_cancorr_nulldist")[0] for i in fpairnames]

        # overarching directory storing data aggregated over worker outputs
        pt_arch_dir = os.path.join(output_basedir,decomp_method+"_permtests_" + date_label)
        if not os.path.isdir(pt_arch_dir):
            os.makedirs(pt_arch_dir)
            print("WARNING: Created directory " + pt_arch_dir)

        # saves aggregated data in overarching directory
        for i in range(len(pairnames)):
            ld_pt.save_permtests(pt_arch_dir,pairnames[i],agg_data[i])

        return pt_arch_dir

    ## run permutation tests in parallel -- calls all functions defined in this module using semiglobal variables
    pt_arch_dir = run_parallel_permtests(output_basedir,n_workers)
    print("Final results saved to directory " + pt_arch_dir)
    return pt_arch_dir


if __name__=="__main__":
    main(sys.argv)
