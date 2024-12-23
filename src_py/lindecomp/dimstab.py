import os
import sys
import csv
import time
import dill
import datetime
import pull_cancorrs
import numpy as np
import matplotlib.pyplot as plt
import perm as rldp
import brainrep as br
import analyze_CCA_permtests as anl_pt

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import HCP_utils as hutils

def main(argvals):
    ## assigning named variables to user inputs
    # filepath to list of names of datasets
    dataset_fname=argvals[1]
    if ".csv" in dataset_fname:
        reps = hutils.load_reduced_data(dataset_fname)
    elif ".brep" in dataset_fname:
        with open(dataset_fname,'rb') as fin:
            reps = dill.load(fin)
    else:
        sys.exit("Unrecognized file extension for brain representation data.")
    ## debug code
    print("Loaded brain reps data from filepath:")
    print(argvals[1])

    # base directory in output tree
    output_basedir=argvals[2]
    ## debug code
    print("Accepted output base directory:")
    print(argvals[2])

    # path to saved, precomputed list of (allowed) permuations
    permpath=argvals[3]
    # read in set of heteroscedastic permutations; convert to int type for indexing; subtract 1 for index differences with matlab
    permset = np.loadtxt(permpath,delimiter=",").astype(int) - 1
    ## debug code
    print("Read in permutation set: showing " + str(permset.shape[1]) + " entries.")


    # path to list of brain representation names (same order as datasets are listed in dataset_fname)
    listpath=argvals[4]
    namelist=hutils.load_namelist(listpath)
    ## debug code
    print("Namelist read:")
    print(namelist)

    # path to list of paired [pairname, optimal_regularization] values
    regpath=argvals[5]
    # read in array of regularization values to test
    regval_list=np.loadtxt(regpath,delimiter=",")
    ## debug code
    print("Regularization values accepted:")
    print(regval_list)

    # linear decomposition algorithm: currently either CCA or PLS (far more likely to be CCA)
    decomp_method=argvals[6]
    ## debug code
    print("Decomp method:")
    print(argvals[6])

    # number of threads requested by job
    if len(argvals) < 7: 
        n_workers = 1
    else:
        n_workers = int(argvals[7])

    ## debug code
    print("Number of parallel workers:")
    print(int(n_workers))

    run_dimensional_stability_tst(reps,output_basedir,permset,namelist,regval_list,decomp_method,n_workers)

def make_reglist(namelist, regval):
    n_names = len(namelist)
    pairnamelist = [namelist[i] + "_and_" + namelist[j] for i in range(n_names) for j in range(i+1,n_names)]
    
    n_pairs = int(n_names*(n_names-1)/2)
    regvals = [regval]*n_pairs

    reglist = [pairnamelist,regvals]
    return reglist

def run_dimensional_stability_tst(reps,output_basedir,permset,namelist,regval_list,decomp_method,n_workers):

    n_regs = len(regval_list)
    dim_vs_reg_list = [None]*n_regs

    for i in range(n_regs):
        regval = regval_list[i]
        reglist_i = make_reglist(namelist,regval)
        ## debug code
        # print("Reglist " + str(i))
        # print(reglist_i)

        regval_tag = "lambda_reg_1e" + str(int(np.log10(regval)))

        data_outdir = os.path.join(output_basedir, "datasets", regval_tag)
        nulldist_outdir = os.path.join(output_basedir, "null_dists", regval_tag)
        cancorrplots_outdir = os.path.join(output_basedir, "cancorr_plots", regval_tag)
        hutils.check_to_make_dirs([data_outdir,nulldist_outdir,cancorrplots_outdir])

        br.pairwise_lindecomp(data_outdir, reps, namelist, reglist=reglist_i, decomp_method=decomp_method)
        pull_cancorrs.pull_res_cancorrs(data_outdir)
        nulldist_outdir = rldp.run_permutation_testing(reps, nulldist_outdir, permset, namelist, reglist_i, decomp_method, n_workers)
        dim_vs_reg_list[i] = anl_pt.visualize_permtests(cancorrplots_outdir, data_outdir, nulldist_outdir, reglist_i)

    dimplot_dir = os.path.join(output_basedir,"dim_plots")
    make_ndim_vs_reg_plots(dimplot_dir,dim_vs_reg_list,regval_list)
        
def make_ndim_vs_reg_plots(output_basedir,dim_vs_reg_list,regval_list):
    fig,ax = anl_pt.plot_ndim_vs_reg(dim_vs_reg_list,np.asarray(regval_list))
    
    date_label = datetime.date.today().strftime("%Y%b%d")
    gen_savename = "ndim_vs_reg_" + date_label

    plot_savename = os.path.join(output_basedir, gen_savename+"_plot.png")
    fig.savefig(plot_savename, dpi=600, bbox_inches='tight')
    
    list_savename = os.path.join(output_basedir, gen_savename+"_list.csv")
    with open(list_savename,'w') as fout:
        write = csv.writer(fout)
        write.writerows(dim_vs_reg_list)



if __name__=="__main__":
    main(sys.argv)
