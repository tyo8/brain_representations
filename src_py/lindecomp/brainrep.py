import os
import re
import csv
import sys
import dill
import numpy as np
import rcca as rcca
import sklearn.cross_decomposition as decomp

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import HCP_utils as hutils


def main(argvals):
    dataset_fname = argvals[1]
    output_basedir = argvals[2]
    namelist_path = argvals[3]
    reglist_path = argvals[4]
    decomp_method = argvals[5]


    reps = hutils.load_reps(dataset_fname)
    namelist = hutils.extract_namelist(namelist_path)
    with open(reglist_path,'r') as fin:
        reglist = list(csv.reader(fin))
        ## debug code
        print(reglist)

    list_path = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/BR_label_list.csv"
    pairwise_lindecomp(output_basedir, reps, namelist, reglist=reglist,
            decomp_method=decomp_method, write_mode=True, param_search=False)


def pairwise_lindecomp(output_basedir, reps, namelist, reglist=[],  
        decomp_method='CCA', write_mode=True, param_search=False):
    n_reps = len(reps)
    lindecomp = switch(decomp_method)
    resvars = [[None for j in range(i+1,n_reps)] for i in range(n_reps)]

    for i in range(n_reps):
        for j in range(i+1,n_reps):
            pairname = namelist[i] + "_and_" + namelist[j]
            X = reps[i]
            Y = reps[j]

            if param_search:
                CCA_res = lindecomp(X,Y,param_search=True)
            else:
                reg_val = find_reg_val(reglist,namelist[i],namelist[j])
                CCA_res = lindecomp(X,Y,reg_val=reg_val,param_search=False)

            if write_mode:
                save_results(CCA_res,output_basedir,namelist[i],namelist[j],decomp_method)
            else:
                resvars[i][j-i-1] = CCA_res

    res_vars = []
    for i in range(len(res_vars)):
        res_vars += resvars[i]
    return res_vars


# given a pair X and Y of NxP and NxQ matrices, computes the CCA between them.
def comp_CCA(X, Y, param_search=True, reg_val=1, comp_prop=0.1):

    Rx,Vx = hutils.svd_reduce(X)
    Ry,Vy = hutils.svd_reduce(Y)

    if reg_val == 0:                     # if regularization hyperparameter is 0, impose simple regularization via PCA reduction
        n_dims = min(round(Rx.shape[0]*comp_prop),  # no. of included PCA components is proportional to no. of subjects; default prop is 1/10
                Rx.shape[1], Ry.shape[1])
        Rx = Rx[:,range(n_dims)]
        Vx = Vx[:,range(n_dims)]
        Ry = Ry[:,range(n_dims)]
        Vy = Vy[:,range(n_dims)]

    n_comps = int(np.amin([Rx.shape, Ry.shape]))-1      # projection dimension (rank) of the CCA
    if param_search:
        reg_vals = np.logspace(-4,7,12)  # learn optimal regularization hyperparameter
        ccaCV = rcca.CCACrossValidate(
                regs = reg_vals,
                numCCs = [n_comps],
                kernelcca = False,
                verbose = False,
                numCV = 5,
                select = 1.)

        ccaCV.train([Rx,Ry])
        ccaCV.validate([Rx,Ry])
        CCA_res = ccaCV

    else:
        CCA_res = rcca.CCA(
                reg = reg_val,
                numCC = n_comps,
                kernelcca = False,
                verbose = False)
        CCA_res.train([Rx,Ry])


    # the following conversion follows from the equations Ux = X @ Ax', X = Rx @ Vx', and Ax = Vx @ Ar,
    # where U_ is the canonical component and A_ are the canonical coefficients.
    CCA_res.ws[0] = Vx @ CCA_res.ws[0] 
    CCA_res.ws[1] = Vy @ CCA_res.ws[1]

    X_cp = cross_predict(CCA_res.comps[1],CCA_res.ws[0])
    Y_cp = cross_predict(CCA_res.comps[0],CCA_res.ws[1])

    cross_pred = [X_cp, Y_cp]

    ## debug code:
    # print("X data dimensions:" + str(X.shape))
    # print("Wx data dimensions:" + str(ccaCV.ws[0].shape))
    # print("Y data dimensions:" + str(Y.shape))
    # print("Wy data dimensions:" + str(ccaCV.ws[1].shape))

    return CCA_res, cross_pred

# given a pair X and Y of NxP and NxQ matrices, computes their PLS regression.
def comp_PLS(X,Y,param_search=False,reg_val=1):
    n_comps = int(np.sqrt(np.amin([X.shape, Y.shape])))         # projection dimension (rank) of the PLS

    RX,VX = hutils.svd_reduce(X)
    RY,VY = hutils.svd_reduce(Y)

    PLS_model = decomp.PLSRegression(n_components = n_comps)    # problem parameterization of PLS regression
    
    if param_search:    # placeholder code for parameter searching (no regularization available for PLS in this version)
        X_scores = None
        Y_scores = None
        PLS_model = None
    else:
        PLS_model.fit(RX,RY)
        X_scores, Y_scores = PLS_model.transform(RX,RY)

    results = [X_scores, Y_scores, PLS_model]
    return results


def cross_predict(comps_Y,weights_X):
    ## debug code:
    # print("Uy dims: " + str(comps_Y.shape))
    # print("Wx dims:" + str(weights_X.shape))

    Wx_inv = np.linalg.pinv(weights_X)          # pseudoinverse of weights/loadings/coefficients (linear map sending data to canonical space)
    X_cp = comps_Y @ Wx_inv                     # cross-predicted value of X, given by X_cp = Y @ Wy' @ (Wx^-1) = Uy @ (Wx^-1)

    return X_cp


# takes in paired list of the form [[variable pair names], [regularization hyperparameters]] 
# and returns the regularization hyperparameter paired to "xname_and_yname"
def find_reg_val(reg_list,xname,yname):
    if reg_list:
        pairname = xname + "_and_" + yname
        idx = reg_list[0].index(pairname)
        reg_val = float(reg_list[1][idx])
    else:
        reg_val = 0
    ## debug code:
    # print("Regularization hyperparameter = " + str(reg_val) + " for " + pairname + ".")

    return reg_val


def save_results(results,output_dir,xname,yname,decomp_method):
    ext   = "." + decomp_method + "_res"
    fname = xname + "_and_" + yname + ext
    fpath = os.path.join(output_dir,fname)

    with open(fpath,'wb') as results_fpath:
        dill.dump(results,results_fpath,protocol = 4)


def switch(argument):
    switcher = {
        "CCA": comp_CCA,
        "PLS": comp_PLS,
     }
    parser = switcher.get(argument, lambda argument: print("Unknown linear decomposition method: "+argument))
    return parser


if __name__=="__main__":
    main(sys.argv)
