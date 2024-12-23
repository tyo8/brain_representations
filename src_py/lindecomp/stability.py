import os
import csv
import sys
import time
import dill
import shutil
import inspect
import datetime
import numpy as np
import brainrep as br

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import HCP_utils as hutils

list_path = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/BR_label_list.csv"

## distributes, runs, and collects results from a stability analysis of regularized CCA over many iterations; manages calls to and options for## "run_stability_iters."
def main(argvals):
    # takes in (global) function arguments & parameters
    dataset_listname = argvals[1]           # path to .csv list of filepaths of list of brain representation filepaths
    output_basedir = argvals[2]             # base directory for output 
    listpath = argvals[3]                   # path to list of CCA variable names
    decomp_method = argvals[4]              # linear decomposition method; default 'CCA', but 'PLS' is also supported
    if len(argvals) < 6:
        par = False                         # no parallelization if number of workers is not specified in job submission
    else:
        par = True                          # parallelization flag
        n_workers = int(argvals[5])         # number of parallel workers specified in job submission
    
    ## internal function to allow worker parameters as globals; implements distributed call to "run_stability_iters"
    def par_stbl_iters(output_bdir):
        brainreps_data = hutils.load_reps(dataset_listname)              # input datasets for pairwise CCA
        run_stability_iters(output_bdir, listpath=listpath, decomp_method=decomp_method,
                read_mode=False, reps=brainreps_data,write_mode=False)      # calls "run_stability_iters" to specific output directory, taking
                                                                            # remaining arguments as globals

    ## initializes parallelization scheme and distributes calls to parallel workers
    def run_parallel_iters(output_basedir,n_workers):
        import pathos.multiprocessing as mp                                 # if parallelization, import multiprocessing library
        date_label = datetime.date.today().strftime("%Y%b%d")               # datestamp for save files

        dirlocs = [os.path.join(output_basedir,                             # list of base directories, one for each worker
            decomp_method+"_stbl_" + str(k) +"_" + date_label) for k in range(n_workers)]

        with mp.ProcessingPool(n_workers) as P:                                       
            P.map(par_stbl_iters,dirlocs)                                   # calls workers to distributed directories and "par_stbl_iters"

        decomp_vals = aggregate_decomps(dirlocs)                            # aggregates data over all distributed directories
        stbl_arch_dir = os.path.join(output_basedir,
                decomp_method+"_stbl_paragg_" + date_label)                 # names archdirectory for aggregated worker data 

        if not os.path.isdir(stbl_arch_dir):
            os.makedirs(stbl_arch_dir)                                      # creates archdirectory if it does not exist
            print("WARNING: Created directory " + stbl_arch_dir)

        return decomp_vals,stbl_arch_dir                                    # returns aggregated worker data and path to archdirectory

    # evaluates par flag
    if par:
        decomp_vals,stbl_arch_dir = run_parallel_iters(output_basedir,n_workers)        # runs parallelized iterations; returns outputs
        save_stability_decomps(decomp_vals,stbl_arch_dir,decomp_method)                 # saves over-iter dataset
        summarize_iters(decomp_vals, output_basedir, decomp_method=decomp_method)       # computes summaries of aggregated worker data

    else:
        stability_testing(output_basedir, dataset_list_name=dataset_listname,
                listpath=listpath, decomp_method=decomp_method)                         # no parallelization


## Runs, collects, and saves stripped CCA results (optimum regularizer, canonical correlations, and canonical correlations)
## for a list of regularization hyperparameters over different realizations of cross validation folds; assesses stability of
## hyperparameter optimization in the rCCA algorithm.
##
## INPUTS:
## --output_basedir:    base directory which stores aggregated and summarized CCA data from all such stability tests in this genre
## --reps:              brain representations (datasaets) on which to pairwise perform CCA; must be nonempty if "dataset_list_name" is empty
## --dataset_list_name: path to .csv list of filepaths of list of brain representation filepaths; must be nonempty if "reps" is empty
## --listpath:          path to .csv list of names of brain representations; default is global "list_path" defined at top of module
## --decomp_method:     linear decomposition algorithm, specified as a string; default 'CCA', but 'PLS' is also (nominally) supported
## --n_iter:            number of iterations (i.e., cross validation realizations); default is 50 (assumed to be a single par-worker's task)
## --chkpt:             flag to checkpoint after every 10 iterations; default is True
## --read_mode:         flag to read variables in from file; if "read_mode"=True, "dataset_list_name" must be nonempty; default is True
## --write_mode:        flag to write intermediate CCA variables to file and read in from disk; default is False
def stability_testing(output_basedir, reps=[], dataset_list_name='',
        listpath=list_path, decomp_method='CCA', n_iter=3, chkpt=True,
        read_mode = True): 

    # computes pairwise linear decomposition of list of brain representations
    decomp_vals = run_stability_iters(output_basedir, dataset_list_name=dataset_list_name,
            listpath=list_path, decomp_method=decomp_method, n_iter=n_iter, chkpt=chkpt,
            read_mode = read_mode, reps = reps)

    # summarizes linear decomposition results over set of iterations
    summarize_iters(decomp_vals, output_basedir, decomp_method=decomp_method)

def run_stability_iters(output_basedir, reps=[], dataset_list_name='',
        listpath=list_path, decomp_method='CCA', n_iter=50, chkpt=True, read_mode=True, write_mode=False):

    var_output_dir = os.path.join(output_basedir,decomp_method + '_vars')       # filepath to output directory 
    if not os.path.isdir(var_output_dir):
        os.makedirs(var_output_dir)                                             # create output directory if does not exist
        print("WARNING: Created directory " + var_output_dir)

    strip = strip_switch(decomp_method)                                         # select correct strip function (based on lindecomp method)
    ext = '.' + decomp_method + '_res'                                          # savefile extension

    if read_mode:
        reps = hutils.load_reps(dataset_list_name)                           # loads brain representations here if in read mode

    namelist = hutils.load_namelist(listpath)                                # list of variable names
    pairnamelist = [namelist[i] + '_and_' + namelist[j]                         # list of pairs of variable names
        for i in range(len(namelist))
        for j in range(i + 1, len(namelist))]

    fpairnamelist = [i+ext for i in pairnamelist]                               # list of filenames for variable pairs
    n_pairs = len(fpairnamelist)                                                # number of variable pairs

    decomp_vals = [[fpairnamelist[j] for i in range(1)] for j in range(len(fpairnamelist))]     # initializes list of decomp values
    elapsed = [None]*n_iter                                                     # time benchmarking

    lindecomp = br.switch(decomp_method)

    for i in range(n_iter):
        tic = time.time()   # benchmarking

        # in theory, it could be helpful to write out intermediate CCA results: in practice, you probably want write_mode=False.
        if write_mode:
            lindecomp(var_output_dir, reps, namelist, reglist=False,
                    decomp_method=decomp_method, write_mode=True, param_search=True)

            res_vars = [None]*n_pairs   # holds CCA results
            for j in range(n_pairs):
                loadname = os.path.join(var_output_dir,fpairnamelist[j])    # path to CCA (stability) results of given pair
                with open(loadname,'rb') as f_in:
                    res_vars[j] = dill.load(f_in)                           # CCA (stability) results for given pair
        else:
            res_vars = lindecomp(var_output_dir, reps, namelist, reglist=False,
                    decomp_method=decomp_method, write_mode=False, param_search=True)   # CCA (stability) results for all pairs 
            
        for j in range(n_pairs):
            stripped_vars = strip(res_vars[j])      # bare essentials of CCA results
            decomp_vals[j].append(stripped_vars)    # list of CCA results; to be exracted from later

        elapsed[i] = time.time() - tic              # benchmarking iteration times
        if chkpt:
            if np.mod(i+1,10) == 0:                 # saves every 10 iterations (about 8 hours)
                save_stability_decomps(decomp_vals,output_basedir,decomp_method)

    save_stability_decomps(decomp_vals,output_basedir,decomp_method) 
    ## Benchmarking:
    print("")
    print("")
    print("BENCHMARKING:")
    # speed benchmarking:
    speed_benchmark(elapsed)


## summarizes statistics of (stripped) CCA outputs 
def summarize_iters(decomp_vals,output_basedir,decomp_method='CCA'):

    summarizer = summ_switch(decomp_method)                     # selects function to summarize decomp list
    stability_summary = summarizer(decomp_vals)                 # summarizes decomp list

    date_label = datetime.date.today().strftime("%Y%b%d")       # date label to tag savefiles and paths
    savename = "stability_summary_" + date_label + ".summ"      # savename for summaries
    saveloc = os.path.join(output_basedir,savename)             # savepath for summaries

    with open(saveloc,'wb') as f_out:
        dill.dump(stability_summary,f_out)                      # saves summaries to savepath "saveloc"

    return saveloc                                              # returns summary savepath


## chooses strip function based on lindecomp method
def strip_switch(argument):
    switcher = {
        "CCA": strip_CCA,                                       # CCA variable strip function
        "PLS": strip_PLS,                                       # PLS variable strip function
     }
    # selects correct switch function: prints message if unrecognized lindecomp method
    strip = switcher.get(argument, lambda argument: print("Unknown linear decomposition method: "+argument))
    return strip

## custom class storing bare-bones CCA outputs
class CCA_stripped_vars:
    def __init__(self,CCAres): 
        self.can_comps = CCAres.comps               # canonical component matrix
        self.can_corrs = CCAres.cancorrs            # canonical correlation vector
        self.lambda_opt = CCAres.best_reg           # optimal regularization hyperparameter

# placeholder code
def strip_PLS(PLSres):
    return None

## strips CCA to bare-bones outputs
def strip_CCA(CCAres):
    ## debug code:
    # print(dir(CCAres))
    strip_res = CCA_stripped_vars(CCAres)           # initializes instance of CCA_res object, a streamlined version of default rCCA output obj
    return strip_res

## chooses summarizer function based on lindecomp method
def summ_switch(argument):
    switcher = {
        "CCA": summ_CCA,                            # CCA variable summarizer
        "PLS": summ_PLS,                            # PLS variable summarizer
     }
    # selects correct data summarizer: prints message if unrecognized type of linear decomp method
    parser = switcher.get(argument, lambda argument: print("Unknown linear decomposition method: "+argument))
    return parser

## custom class to store over-iteration distributions of CCA quantities of interest
class CCA_Stability:
    def __init__(self,name,lambda_opt,comp_dists_spectral,comp_dists_Frobenius,
            comp_corrs,comp_snorms,comp_Fnorms,corr_dists,corr_summaries):
        self.data_pair_name = name                          # name of the pair of brain reps with data (pairname)
        self.lambda_opt = lambda_opt                        # optimal regularization hyperparameter over CV folds
        self.comp_dists_spectral = comp_dists_spectral      # pairwise spectral dist b/t canonical component matrices over iters
        self.comp_dists_Frobenius = comp_dists_Frobenius    # pairwise Frobenius dist b/t canonical component matrices over iters
        self.comp_corrs = comp_corrs                        # pairwise correlations b/t flattened canonical component matrices over iters
        self.comp_norms_spectral = comp_snorms              # spectral norm of canonical component matrices from each iter
        self.comp_norms_Frobenius = comp_Fnorms             # Frobenius norm of canonical component matrices from each iter
        self.corr_dists = corr_dists                        # pairwise L2 distance between canonical correlation vectors over iters
        self.corr_summaries = corr_summaries                # min, mean, and max of canonical correlation vectors from each iter

# placeholder code
def summ_PLS(PLS_stbl):
    return None

## collects computed summaries of CCA variables over all iterations
def summ_CCA(CCA_stbl):
    n_vars       = len(inspect.signature(CCA_Stability.__init__).parameters)-1  # number of summary measures collected per iteration set
    summary_vals = [None]*n_vars                                                # initialization of summary variables

    n_pairs = len(CCA_stbl)                                                     # number of brain rep pairs
    all_summaries = [None]*n_pairs                                              # intialization of collection of summary variables

    for i in range(n_pairs):
        summary_vals[0] = CCA_stbl[i][0]                                        # collect pairname
        summary_vals[1:] = summarize_CCA_vars(CCA_stbl[i][1:])                  # collect CCA variables to summarize

        cca_summ = CCA_Stability(*summary_vals)                                 # summarize CCA variables
        all_summaries[i] = cca_summ                                             # add summary variables to collection

    return all_summaries                                                        # return collection of summaries

## computes summaries of CCA over-iteration results variable set
def summarize_CCA_vars(CCA_res_list):
    n_iter = len(CCA_res_list)                          # number of iterations

    can_corrs = [None]*n_iter                           # initialize canonical correlations list
    U_comps = [None]*n_iter                             # initialize canonical components (X) list
    V_comps = [None]*n_iter                             # initialize canonical components (Y) list
    lambda_opt = [None]*n_iter                          # initialize optimal regularization hyperparameter list
    for i in range(n_iter):
        U_comps[i] = CCA_res_list[i].can_comps[0]       # collect i-th canonical component matrix (X)
        V_comps[i] = CCA_res_list[i].can_comps[1]       # collect i-th canonical component matrix (Y)
        can_corrs[i] = CCA_res_list[i].can_corrs        # collect i-th vector of canonical correlations
        lambda_opt[i] = CCA_res_list[i].lambda_opt      # collect i-th optimal regularization hyperparameter

    comps = [U_comps,V_comps]                           # pair X and Y canonical component lists in a single list

    comp_dists_spectral,comp_dists_Frob,comp_corrs,comp_snorms,comp_Fnorms = compare_components(comps)  # compute matrix distance metrics
    corr_dists,corr_summaries = compare_cancorrs(can_corrs)                                             # compute vector distances & summaries

    summ_list = [lambda_opt,comp_dists_spectral,comp_dists_Frob,comp_corrs,
            comp_snorms,comp_Fnorms,corr_dists,corr_summaries]                                          # collect summaries into list
    return summ_list                                                                                    # return summary list

## collects pairwise distance metrics between component matrices over all CCA sets
def compare_components(components):
    n_sets = len(components)                # number of variable sets over which CCAs were computed (e.g., allows for multiset CCA)
    comp_dists_spectral = [None]*n_sets     # initializes spectral distance collection
    comp_dists_Frob = [None]*n_sets         # initializes spectral distance collection
    comp_corrs = [None]*n_set               # initializes spectral distance collections
    comp_snorms = [None]*n_sets             # initializes spectral distance collection
    comp_Fnorms = [None]*n_sets             # initializes spectral distance collection

    for i in range(n_sets):
        # computes several distance metrics between component matrices
        comp_dists_spectral[i],comp_dists_Frob[i],comp_corrs[i],comp_snorms[i],comp_Fnorms[i] = compare_matrices(components[i])
        
    return comp_dists_spectral,comp_dists_Frob,comp_corrs,comp_snorms,comp_Fnorms                       # return pairwise distances

## computes comparisons between and summaries of canonical component matrices from different iterations
def compare_matrices(mtx_list):
    n_iter = len(mtx_list)

    Gram_spectral = np.zeros((n_iter,n_iter))       # initialize spectral distance (Gram) matrix
    Norm_spectral = np.zeros((n_iter,))             # initialize spectral norm vector
    Gram_Frobenius = np.zeros((n_iter,n_iter))      # initialize Frobenius distance (Gram) matrix
    Norm_Frobenius = np.zeros((n_iter,))            # initialize Frobenius norm vector
    corr_matrix = np.zeros((n_iter,n_iter))         # initialize pairwise correlation matrix

    for i in range(n_iter):
        Norm_spectral[i] = np.linalg.norm( mtx_list[i] , ord=2 )                            # compute i-th spectral norm
        Norm_Frobenius[i] = np.linalg.norm( mtx_list[i] , ord='fro' )                       # compute i-th Frobenius norm
        for j in range(i+1,n_iter):
            Gram_spectral[i,j] = np.linalg.norm( mtx_list[i]-mtx_list[j] , ord=2)           # compute ij-th spectral distance
            Gram_Frobenius[i,j] = np.linalg.norm( mtx_list[i]-mtx_list[j] , ord='fro')      # compute ij-th Frobenius distance
            corr_matrix[i,j] = matrix_corr(mtx_list[i],mtx_list[j])                         # compute ij-th correlation of componenet matrices

    # fill in lower left triangle of symmetric distance matrices
    Gram_spectral += np.matrix.transpose(Gram_spectral)
    Gram_Frobenius += np.matrix.transpose(Gram_Frobenius)
    corr_matrix += np.matrix.transpose(corr_matrix)

    return Gram_spectral,Gram_Frobenius,corr_matrix,Norm_spectral,Norm_Frobenius            # return norm and distance matrices

## computes matrix correlation
def matrix_corr(M1,M2):
    M1f = M1.flatten('F')   # flattens matrix into vector
    M2f = M2.flatten('F')   # (uses column-first flattening)

    mini_corr_mtx = np.corrcoef(M1f,M2f)        # 2x2 np.corrcoef matrix [AA AB; BA BB]
    corr_val = mini_corr_mtx[0,1]               # selects only nontrivial corrcoef value
    return corr_val

## computes comparisons between and summaries of canonical correlations from different iterations
def compare_cancorrs(cancorrs):
    n_iter = len(cancorrs)                      # number of canonical correlations/components

    cancorr_dists = np.zeros((n_iter,n_iter))   # initializes L2 Gram matrix between canonical correlations
    cancorr_min = [None]*n_iter                 # initializes list of min corr values
    cancorr_mean = [None]*n_iter                # initializes list of mean corr values
    cancorr_max = [None]*n_iter                 # initializes list of max corr values

    for i in range(n_iter):
        cancorr_min[i] = np.amin(cancorrs[i])                       # minimum canonical correlation
        corrvals = np.abs(cancorrs[i][np.nonzero(cancorrs[i])])     # replace canonical correlations with their absolute values
        cancorr_mean[i] = np.exp(np.mean(np.log(corrvals)))         # geometric mean of canonical correlations
        cancorr_max[i] = np.amax(cancorrs[i])                       # maximum canonical correlation
        for j in range(i+1,n_iter):
            cancorr_dists[i,j] = np.mean(np.abs(cancorrs[i]) - np.abs(cancorrs[j]))     # difference of magnitudes of correlations (L2 Gram)

    cancorr_dists += np.matrix.transpose(cancorr_dists)             # fill in lower left traingle of symmetric L2 Gram matrix
    cancorr_summaries = [cancorr_min,cancorr_mean,cancorr_max]      # collect correlation vector summaries in list
    return cancorr_dists,cancorr_summaries                          # return correlation distances and summaries


## saves over-iteration collection of linear decomp variables
def save_stability_decomps(decomp_vals,output_dir,decomp_method):
    N = len(decomp_vals)                                        # number of pairs of CCA variables

    ext = "." + decomp_method + "_stbl"                         # file extension

    for i in range(N):
        pairname = decomp_vals[i][0].split('.')[0]              # name of pair of CCA variables
        saveloc = os.path.join(output_dir,pairname + ext)       # filepath to save collection of over-iter variables
        # print(pairname + " stability variables saved to " + saveloc)
        data = decomp_vals[i][1:]                               # over-iter variables

        with open(saveloc,'wb') as f_out:
            dill.dump(data,f_out)                               # saves over-iter variables


## aggregates over-iter collections of linear decomp variables from different sessions from different directories
def aggregate_decomps(dirlocs, del_dir=True):

    ## debug code:
    # print(dirlocs)

    fpairnames = [fn for fn in os.listdir(dirlocs[0]) if ".CCA_stbl" in fn]     # collects list of files saving over-iter collections
    n_pairs = len(fpairnames)                                                   # number of variable pairs
    decomp_vals = [[fpairnames[i]] for i in range(n_pairs)]                     # initializes list of lists of over-iter variables
                                                                                #   (first element of list is pairname)
    ## debug code:
    # print(fpairnames)

    for i in range(n_pairs):
        for j in dirlocs:
            dataloc = os.path.join(j,fpairnames[i])                             # filepath to over-iter variable list for i-th variable pair
            with open(dataloc,'rb') as fin:
                data = dill.load(fin)                                           # loads data from filepath

            decomp_vals[i] += data                                              # join loaded data list to over-iter variables list

    if del_dir:
        for i in dirlocs:
            shutil.rmtree(i)                                                    # removes directory tree whose data has now been aggregated

    return decomp_vals                                                          # returns aggregated data


## BENCHMARKING
## speed benchmarking
def speed_benchmark(elapsed):
    n_iter = len(elapsed)
    print("Total within-iteration time: " + str(np.round(sum(elapsed))) + " sec")
    print("Average iteration length: " + str(np.mean(elapsed)) + " sec")
    print("Std. dev of iteration time: " + str(np.sqrt(np.var(elapsed))) + " sec")
    print("Number of iterations: " + str(n_iter))
    print("")


if __name__=="__main__":
    main(sys.argv)
