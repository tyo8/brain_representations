import os
import sys
import csv
import dill
import copy
import HCP_utils
import numpy as np
import phom_figs as pf
import comp_Psim_mtx as cPm
import scipy.special as scsp
import matplotlib.pyplot as plt
import SubjSubjHomologies as SSH
import lindecomp_brainrep as ld_br

## summary of module:
## 
## 1) pull and load brainrep datasets (full, not yet SVD'd)
## 2) run all CCAs and retain outputs (in particular, weight and canonical component matrices)
## 3) while looping through (user-defined or hard coded?) component weight numbers K:
##       3a) truncate weight/component matrices after K components
##       3b) compute cross-predicted approximations (X~,Y~) of each brainrep variable in data pair
##       3c) compute 4 Pearson-dissimilary Gram matrices, for X~ and X-X~, Y~ and Y-Y~
##       3d) compute persistent homologies from each Gram matrix
##       3e) compute Betti curves from each persistence module 
## 4) Group and save decomp-grouped Betti curve plots
##       4a) show each brain rep with all of its data pairs; show X~ and X-X~ on the same plot
##       4b) make new plot for each choice of component rank and each brainrep type; should be (#brainrep)x(#comp_ranks) plots in total
## 5) Group and save pairwise Betti curve plots
##       5a) for a given pair (X,Y) of brain reps, show X and Y, X~ and Y~, and X-X~ and Y-Y~ on the same plot
##       5b) make new plot for each choice of component rank and each brainrep pair
##       5c) in addition, make plot for each pair that also shows all ranks

def main(argvals):
    dataset_list_name = argvals[1]
    reps = HCP_utils.load_reps(dataset_list_name)

    output_basedir = argvals[2]

    namelist_path = "/scratch/tyoeasley/brain_representations/BR_label_list.csv"
    namelist = HCP_utils.load_namelist(namelist_path)

    reglist_path = argvals[3]
    with open(reglist_path,'r') as fin:
        reglist = list(csv.reader(fin))
    
    decomp_method = argvals[4]
    comp_rank = [1000, 500, 100, 50, 10, 5]

    if len(argvals) > 5:
        hom_dims = argvals[5]
    else:
        hom_dims = (0,1,2)

    # compute CCAs between all representation pairs
    full_lindecomps = iter_lindecomp(reps, reglist=reglist, namelist=namelist, decomp_method=decomp_method)
    # save CCA results set
    save_intm_data(output_basedir, "full_lindecomps.nlist", full_lindecomps)
    
    make_bcurves_over_rank(full_lindecomps, comp_rank, output_basedir, hom_dims=hom_dims) 


## iteratively computes Betti curves from rank-restricted CCA cross-predictions (and their residuals) over all ranks of interest
def make_bcurves_over_rank(full_lindecomps, comp_rank, output_basedir, hom_dims=(0,1,2)):
    n_subj = full_lindecomps[0][1].shape[0]
    comp_rank = [i for i in comp_rank if i<=n_subj]         # removes approximation ranks larger than CCA rank

    all_bcurves = [None]*len(comp_rank)                     # initialize betti curveset (over all ranks)

    for i in range(len(comp_rank)):
        # base directory for i_th approximation rank
        outdir_i = os.path.join(output_basedir,"rank_"+str(comp_rank[i]))

        if not os.path.isdir(outdir_i):
            os.makedirs(outdir_i)
            print("Warning: created directory " + outdir_i)

        # compute and save Gram matrix at given approximation rank
        approx_reps, psim_grams = filt_lindecomps(full_lindecomps, CCrank=comp_rank[i])
        save_intm_data(outdir_i, "psim_grams.nlist", psim_grams)

        rank_bcurves, all_phoms = iter_comp_bcurves(psim_grams, outdir_i, hom_dims=hom_dims)
        save_intm_data(outdir_i, "phom_data.nlist", rank_bcurves)
        save_intm_data(outdir_i, "bcurves_data.nlist", rank_bcurves)

        decomp_plot_betti_curves(outdir_i, rank_bcurves, CCrank=comp_rank[i])
        pairwise_bcurves_dir = os.path.join(outdir_i, 'pairwise')
        pairwise_plot_rank_bcurves(pairwise_bcurves_dir, rank_bcurves, CCrank=comp_rank[i])

        all_bcurves[i] = rank_bcurves

    save_intm_data(output_basedir, "all_betti_curves.nlists", all_bcurves)        
    pairwise_plot_all_bcurves(all_bcurves, output_basedir, rank_list=comp_rank)


## computes CCA for all pairs of variables in reps
def iter_lindecomp(reps, reglist=[], namelist=[], decomp_method='CCA'):
    n_reps = len(reps)
    lindecomp = ld_br.switch(decomp_method)

    full_lindecomps = [[None,None,[[None for j in range(n_reps-1)],[None for j in range(n_reps-1)]]] for i in range(n_reps)]
    
    print("Computing initial CCA decompositions for all " + str(n_reps) + " methods...")
    for i in range(n_reps):
        X = reps[i]
        full_lindecomps[i][0] = namelist[i]                          # stores name of X variable
        full_lindecomps[i][1] = X                                    # stores X variable
        for j in range(i+1,n_reps):
            full_lindecomps[i][2][0][j-1] = namelist[j]              # stores name of Y variable in list of X decompositions
            full_lindecomps[j][2][0][i] = namelist[i]                # stores name of X variable in list of Y decompositions
            Y = reps[j]

            reg_val = ld_br.find_reg_val(reglist,namelist[i],namelist[j])                   # regularization hyperparameter
            CCA_res,cross_pred = lindecomp(X, Y, param_search=False, reg_val = reg_val)     # CCA results 

            comps = CCA_res.comps   # CCA components Ux and Uy
            ws = CCA_res.ws         # CCA weights/loadings Wx and Wy
            
            full_lindecomps[i][2][1][j-1] = [comps[1], ws[0]]        # Uy and Wx
            full_lindecomps[j][2][1][i] = [comps[0], ws[1]]          # Ux and Wy

    print("done.")
    return full_lindecomps


## computes rank-restricted CCA cross-predictions (and corresponding Pearson-based Gram matrices) from full CCAs and rank value
def filt_lindecomps(full_lindecomps, CCrank=1000):
    approx_reps = copy.deepcopy(full_lindecomps)
    psim_grams = copy.deepcopy(full_lindecomps)

    print("Computing cross-predictions from rank-" + str(CCrank) + " CCA data...")
    p=2                                                                 # sim to p-norm parameter; power that rho(X,Y) raised to in dist
    for i in range(len(full_lindecomps)):
        lindecomps_i = full_lindecomps[i][2]
        X = full_lindecomps[i][1]
        psim_grams[i][1] = HCP_utils.p_dist(cPm.comp_Psim_from_mtx(X), order=p)

        for j in range(len(lindecomps_i[0])):
            comp_ij = lindecomps_i[1][j][0]
            ws_ji = lindecomps_i[1][j][1]

            filt_comp_ij = comp_ij[:,:CCrank]
            filt_ws_ji = ws_ji[:,:CCrank]
            X_cp = ld_br.cross_predict(filt_comp_ij, filt_ws_ji)        # cross-predicted X ("shared" information according to CCA)
            Xapp_pGram = HCP_utils.p_dist(cPm.comp_Psim_from_mtx(X_cp),       # Pearson correlation p-"Gram" matrix from CCA-approximated X
                    order=p)
            Xcomp_pGram = HCP_utils.p_dist(cPm.comp_Psim_from_mtx(X - X_cp),  # Pearson correlation p-"Gram" matrix from compliment to Xapprox
                    order=p)  
            approx_reps[i][2][1][j] = [X_cp, X - X_cp]
            psim_grams[i][2][1][j] = [Xapp_pGram, Xcomp_pGram]

    print("done.")
    return approx_reps, psim_grams


## computes persistent homology data and Betti curves from rank-restricted Gram matrices of cross-predictions and their residuals
def iter_comp_bcurves(psim_grams,outdir_i, hom_dims=(0,1,2)):
    n_reps = len(psim_grams)
    
    all_phoms = [[None,None,[None,None,None]] for i in range(n_reps)]
    rank_bcurves = [[None,None,[None,None,None]] for i in range(n_reps)]

    for i in range(n_reps):
        repname = psim_grams[i][0]
        X_pGram = psim_grams[i][1]
        
        all_phoms[i][0] = repname
        rank_bcurves[i][0] = repname

        name_stack = [None]*(n_reps-1)
        approx_stack = [None]*(n_reps-1)
        apcomp_stack = [None]*(n_reps-1)

        for j in range(n_reps-1):
            name_stack[j] = psim_grams[i][2][0][j]
            approx_stack[j] = psim_grams[i][2][1][j][0]
            apcomp_stack[j] = psim_grams[i][2][1][j][1]

        # add name stack to phoms and bcurves
        all_phoms[i][2][0] = name_stack                         
        rank_bcurves[i][2][0] = name_stack                         
        name_stack = [repname] + name_stack
        
        long_stack = [X_pGram] + approx_stack + apcomp_stack
        print("long_stack shape for " + repname + " = "  + str(len(long_stack)) + "x" + str(long_stack[0].shape))

        print("Computing persistence diagrams for all " + str(len(long_stack)) + " kernelizations of " + repname + "...")
        long_phoms = SSH.compute_phom(long_stack, hom_dims=hom_dims)
        print("long_phoms shape for " + repname + " = "  + str(long_phoms.shape))
        # add name stack to persistent homology data 
        all_phoms[i][1] = long_phoms[0,:,:]
        # stack of phom dgms from approximation
        all_phoms[i][2][1] = long_phoms[1:n_reps,:,:]             
        # stack of phom dgms from complement to approximation
        all_phoms[i][2][2] = long_phoms[n_reps:,:,:]

        print("Computing Betti curves for all " + str(len(long_phoms)) + " persistence diagrams...")
        long_bcurves = SSH.comp_Betti_curves(phom_dgms=long_phoms, write_mode=False)
        print("done.")
        print("long_bcurves shape for " + repname + " = "  + str(long_bcurves.shape))
        # betti curves from original data
        rank_bcurves[i][1] = long_bcurves[0,:,:]
        # stack of betti curves from approximation
        rank_bcurves[i][2][1] = np.concatenate((long_bcurves[0:n_reps,:,:], long_bcurves[-1,:,:][None,:,:]), axis=0)
        # stack of betti curves from complement to approximation
        rank_bcurves[i][2][2] = long_bcurves[n_reps:,:,:]


    return rank_bcurves, all_phoms


## plots Betti curves in "decomposition order": i.e., for each representation, shows the Betti curve computed from that representation
## and from all cross-predictions of that variable with each of their residuals
def decomp_plot_betti_curves(output_dir, rank_bcurves, CCrank=1000):
    n_reps = len(rank_bcurves)

    for i in range(n_reps):
        primary_name = rank_bcurves[i][0]
        primary_bcurves = rank_bcurves[i][1]
        bcurve_dirs = os.path.join(output_dir,primary_name)
        print("Betti curves for " + primary_name + " saved to " + bcurve_dirs)
        if not os.path.isdir(bcurve_dirs):
            os.makedirs(bcurve_dirs)
            print("Warning: created directory " + bcurve_dirs)

        secondaries = rank_bcurves[i][2]
        bcurves_names = [primary_name] + secondaries[0]             # collection of paired approximate & complimentary Betti curves 
        bcurves_approx = secondaries[1]                             # Betti curves of Xhat
        bcurves_complement = secondaries[2]                         # Betti curves of X-Xhat

        print("shape of approximate Betti curveset: ", bcurves_approx.shape)
        print("shape of complementary Betti curveset: ", bcurves_complement.shape)

        colorset = plt.cm.rainbow(np.linspace(0,1,n_reps))

        title_line2 = 'shared and unique subspaces of ' + primary_name + ' (rank = ' + str(CCrank) + ')'

        figlist, axlist = pf.export_bcurves(bcurve_dirs, betti_curveset=bcurves_approx[[0,-1],:,:],
                labelset=[bcurves_names[0]], colors=[colorset[0]], linestyle='-',
                title_suffix=title_line2)

        pf.export_bcurves(bcurve_dirs, betti_curveset=bcurves_approx[1:,:,:], 
                parent_figlist=figlist, parent_axlist=axlist,
                labelset=bcurves_names[1:], colors=colorset[1:,:], linestyle='--', 
                title_suffix=title_line2)

        pf.export_bcurves(bcurve_dirs, betti_curveset=bcurves_complement, 
                parent_figlist=figlist, parent_axlist=axlist,
                labelset=bcurves_names[1:], colors=colorset[1:,:], linestyle=':', 
                title_suffix=title_line2)

        plt.close('all')



## plots Betti curves in "pairwise order" at a given rank: i.e., for each pair of representations, shows the Betti curves computed from both 
## representations, both cross-predictions, and both residuals -- produces plots at *each* rank
def pairwise_plot_rank_bcurves(output_dir, rank_bcurves, CCrank=1000):
    n_reps = len(rank_bcurves)

    
    print("Exporting (single-rank) pairwise-grouped Betti curves ...")
    for i in range(n_reps):
        xname = rank_bcurves[i][0]
        X = rank_bcurves[i][1]
        n_dims = X.shape[0]
        for j in range(i+1,n_reps):
            plt.close('all')
            figlist = [None for i in range(n_dims)]
            axlist = [None for i in range(n_dims)]
            for k in range(n_dims):
                fig,ax = plt.subplots()
                figlist[k] = fig
                axlist[k] = ax
            
            yname = rank_bcurves[j][0]
            pairname = xname + '_and_' + yname
            bcurve_dirs = os.path.join(output_dir,pairname)
            title_line2 = pairname + ' (rank = '+str(CCrank)+')'
            
            XfromY_idx = rank_bcurves[i][2][0].index(yname)              # index of the cross-prediction of X from Y (+1 for preappended B(X))
            Xapp_bcurve = rank_bcurves[i][2][1][XfromY_idx+1,:,:]        # Betti curve and filtration parameter values from X cross-pred
            Xres_bcurve = rank_bcurves[i][2][2][[XfromY_idx,-1],:,:]       # "      "       "       "       from residuals of X cross-pred
            Xcurves = np.concatenate((np.stack([X, Xapp_bcurve],
                axis=0), Xres_bcurve), axis=0)
            label1 = [xname, 'X_approx', 'X_resid']
            colorset1 = plt.cm.rainbow(np.linspace(0,1/10,3))

            Y = rank_bcurves[j][1]
            YfromX_idx = rank_bcurves[j][2][0].index(xname)              # index of the cross-prediction of Y from X (+1 for preappended B(X))
            Yapp_bcurve = rank_bcurves[j][2][1][YfromX_idx+1,:,:]        # Betti curve and filtration parameter values from Y cross-pred
            Yres_bcurve = rank_bcurves[j][2][2][[YfromX_idx,-1],:,:]       # "      "       "       "       from residuals of Y cross-pred
            Ycurves = np.concatenate((np.stack([Y, Yapp_bcurve],
                axis=0), Yres_bcurve), axis=0)
            label2 = [yname, 'Y_approx', 'Y_resid']
            colorset2 = plt.cm.rainbow(np.linspace(9/10,1,3))

            axfigset = subplot_rank_bcurves(bcurve_dirs, curveset=Xcurves,
                    axfigset=[figlist, axlist], labelset=label1,
                    colors=colorset1, title_suffix=title_line2)
            
            subplot_rank_bcurves(bcurve_dirs, curveset=Ycurves,
                    axfigset=axfigset, labelset=label2,
                    colors=colorset2, title_suffix=title_line2)
            
            print("shape of each curveset: ", Xcurves.shape)


    print("done.")


def subplot_rank_bcurves(bcurve_dirs, curveset=[], colors=[], 
        axfigset=[None,None], labelset=[], title_suffix=''):
    figlist = axfigset[0]
    axlist = axfigset[1]
    ls_list = ['-','--',':']
    n_curves = len(colors)

    assert n_curves==3, "Expected 3 input curves (orig, approx, resid) to subplot_rnak_bcurves."
    
    for i in range(n_curves):
        figlist,axlist = pf.export_bcurves(bcurve_dirs, betti_curveset=curveset[[i,-1],:,:],
                parent_figlist=figlist, parent_axlist=axlist,
                labelset=[labelset[i]], colors=[colors[i]],
                linestyle=ls_list[i], title_suffix=title_suffix)

    axfigset = [figlist, axlist]

    return axfigset


## plots Betti curves in "pairwise order" *over all ranks*: i.e., for each pair of representations, shows the Betti curves computed from both 
## representations, both cross-predictions, and both residuals -- produces a single plot for each pair, over all ranks
def pairwise_plot_all_bcurves(all_bcurves, output_basedir, rank_list=[]):
    n_ranks = len(all_bcurves)
    n_reps  = len(all_bcurves[0])

    assert n_ranks==len(rank_list), "list of rank values must have same number of entries as list of Betti curve data"

    colornums = np.append(np.append(np.append([0],np.linspace(.1, .25, n_ranks)),
        np.linspace(.75, .9, n_ranks)), [1])
    colorset = plt.cm.rainbow(colornums)

    # colorset_X = plt.cm.rainbow(np.linspace(.1, .25, n_ranks))
    # colorset_Y = plt.cm.rainbow(np.linspace(.75, .9, n_ranks))

    print("Exporting pairwise Betti curves (for each pair) over all ranks...")

    for i in range(n_reps):
        xname = all_bcurves[0][i][0]
        X_bcurve = all_bcurves[0][i][1]
        X_filts = all_bcurves[0][i][2][2][-1,:,:]
        X_bcurveset = np.stack((X_bcurve, X_filts), axis=0)

        for j in range(i+1,n_reps):
            yname = all_bcurves[0][j][0]
            pairname = xname+'_and_'+yname
            print("for", pairname)

            Y_bcurve = all_bcurves[0][j][1]
            Y_filts = all_bcurves[0][j][2][2][-1,:,:]
            Y_bcurveset = np.stack((Y_bcurve, Y_filts), axis=0)

            bcurve_dir = os.path.join(output_basedir, pairname)

            plt.close('all')

            figlist,axlist = pf.export_bcurves(bcurve_dir, betti_curveset=X_bcurveset,
                    labelset=[xname], colors=[colorset[0]], linestyle='-',
                    title_suffix=pairname+', all ranks')

            pf.export_bcurves(bcurve_dir, betti_curveset=Y_bcurveset,
                    parent_figlist=figlist, parent_axlist=axlist,
                    labelset=[yname], colors=[colorset[-1]], linestyle='-',
                    title_suffix=pairname+', all ranks')

            for k in range(n_ranks):
                label = 'rank_'+str(rank_list[k])

                XfromY_kidx = all_bcurves[k][i][2][0].index(yname)
                YfromX_kidx = all_bcurves[k][j][2][0].index(xname)

                Xapp_kbcurve = all_bcurves[k][i][2][1][[XfromY_kidx+1,-1],:,:]        # Betti curve from X cross-pred
                pf.export_bcurves(bcurve_dir, betti_curveset=Xapp_kbcurve,
                        parent_figlist=figlist, parent_axlist=axlist,
                        labelset=[label+'_app'], colors=[colorset[k+1]], linestyle='--',
                        title_suffix=pairname+', all ranks')

                Yapp_kbcurve = all_bcurves[k][j][2][1][[YfromX_kidx+1,-1],:,:]        # Betti curve from Y cross-pred
                pf.export_bcurves(bcurve_dir, betti_curveset=Yapp_kbcurve,
                        parent_figlist=figlist, parent_axlist=axlist,
                        labelset=[None], colors=[colorset[k+1+n_ranks]], linestyle='--',
                        title_suffix=pairname+', all ranks')

                Xres_kbcurve = all_bcurves[k][i][2][2][[XfromY_kidx,-1],:,:]       # Betti curve and filtpars from resids of X cross-pred
                pf.export_bcurves(bcurve_dir, betti_curveset=Xres_kbcurve,
                        parent_figlist=figlist, parent_axlist=axlist,
                        labelset=[None], colors=[colorset[k+1]], linestyle=':',
                        title_suffix=pairname+', all ranks')

                Yres_kbcurve = all_bcurves[k][j][2][2][[YfromX_kidx,-1],:,:]       #   "     "          "     from resids of Y cross-pred
                pf.export_bcurves(bcurve_dir, betti_curveset=Yres_kbcurve,
                        parent_figlist=figlist, parent_axlist=axlist,
                        labelset=[None], colors=[colorset[k+1+n_ranks]], linestyle=':',
                        title_suffix=pairname+', all ranks')

    print("done.")




## saves intermediate data
def save_intm_data(outdir,fname,data):
    if not os.path.isdir(outdir):
        os.path.makedirs(outdir)

    saveloc = os.path.join(outdir,fname)
    print("saving intermediate data to " + saveloc + "...")
    with open(saveloc,'wb') as fout:
        dill.dump(data, fout)

    print("done.")


if __name__=='__main__':
    main(sys.argv)
