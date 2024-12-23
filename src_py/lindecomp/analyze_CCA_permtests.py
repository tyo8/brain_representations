import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection

def main(argvals):
    if len(argvals) > 4:
        reglist_path = argvals[4]
    else:
        reglist_path = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/nonPH_analysis/CCA/reglist_2022Jan02_med.csv"
        reglist_path = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/nonPH_analysis/CCA/reglist0.csv"
    with open(reglist_path,'r') as fin:
        reglist = list(csv.reader(fin))

    visualize_permtests(argvals[1],argvals[2],argvals[3],reglist)

def visualize_permtests(viz_dir, data_dir, null_dir, reglist, prctile_range = (1,99)):
    floc_datalist = [os.path.join(os.path.abspath(data_dir),fname) 
            for fname in os.listdir(data_dir) if "_cancorr_" in fname]
    floc_nulllist = [os.path.join(os.path.abspath(null_dir),fname) 
            for fname in os.listdir(null_dir) if "_cancorr_" in fname]
    fname_datalist = [A.split(os.path.sep)[-1].split("_cancorr_")[0] for A in floc_datalist]
    fname_nulllist = [A.split(os.path.sep)[-1].split("_cancorr_")[0] for A in floc_nulllist]

    n_pairs = len(floc_datalist)
    sig_comps = [fname_datalist,[None]*n_pairs]

    fig_all,ax_all = plt.subplots()
    lastflag = False
    colors = plt.cm.rainbow(np.linspace(0,1,n_pairs))
    for i in range(n_pairs):
        floc_data = floc_datalist[i]
        null_idx = fname_nulllist.index(fname_datalist[i])
        floc_null = floc_nulllist[null_idx]
        
        print("Pair no. " + str(i) + ":")
        print("pairname_data = " + fname_datalist[i])
        print("pairname_null = " + fname_nulllist[null_idx])
        assert fname_datalist[i] == fname_nulllist[null_idx], "You cannot compare correlations from different data pairs."
        
        if i==(n_pairs-1):
            lastflag=True

        fig,ax,n_sig,fig_all,ax_all = plot_cancorrs(floc_data, floc_null, reglist,
                prctile_range = prctile_range,
                parent_fig=fig_all, parent_ax=ax_all, lastflag=lastflag, parent_color=colors[i])
        ax.legend()

        fig_all.axes.append(ax)

        fig.savefig(os.path.join(viz_dir,fname_datalist[i] + "_pt.png"), dpi=600, bbox_inches='tight')

        sig_comps[1][i] = n_sig

    ax_all.legend()
    fig_all.savefig(os.path.join(viz_dir,"all_cases_cancorrs_pt.png"), dpi=600, bbox_inches='tight')

    plt.close('all')

    return sig_comps

def plot_cancorrs(floc_data, floc_null, reglist, prctile_range = (1,99),
        parent_fig=None, parent_ax=None, lastflag=False, parent_color=plt.cm.rainbow(0)):
    
    fig,ax,extreme_vals = plot_null_cancorrs(floc_null,prctile_range=prctile_range)
    
    data = np.loadtxt(floc_data,delimiter=",")
    #debug code:
    print(type(data))
    print(data.shape)
    if extreme_vals:
        data = -np.log10(1-np.abs(data))\

    comps = np.asarray(range(len(data)))

    ax.plot(comps,np.abs(data), 'ro-', label = "data", markersize=1, markerfacecolor="None")
    is_sig,fdr_pvals = find_sigcomps(floc_data,floc_null)
    n_sig = sum(is_sig)
    sig_comps = comps[is_sig]
    sig_data = data[is_sig] 
    ax.plot(sig_comps,np.abs(sig_data), 'go', label=r"$p_{FDR} < 0.05$", markersize=2)

    
    pairname_null = floc_null.split(os.path.sep)[-1].split("_cancorr_")[0]
    reg_val = float(reglist[1][reglist[0].index(pairname_null)])
    fig,ax = format_cancorrplot_axes(pairname_null, fig, ax, reg_val, prctile_range, n_sig, extreme_vals = extreme_vals)

    print("number of significant shared dimensions: " + str(n_sig))
    
    if ((parent_fig is not None) and (parent_ax is not None)):
        if extreme_vals:
            data = 1 - 10**(-data)
            sig_data = data[is_sig]

        parent_ax.plot(comps,np.abs(data), 'ro-', label = pairname_null.replace("_"," "),
                c=parent_color, markersize=1, markerfacecolor="None")
        parent_ax.plot(sig_comps,np.abs(sig_data), 'go', markersize=2)

        if lastflag:
            parent_fig,parent_ax = format_cancorrplot_axes("All_BRep_Pairs", parent_fig, parent_ax, "N/A", prctile_range, n_sig, extreme_vals = False)

    return fig,ax,n_sig,parent_fig,parent_ax

def plot_null_cancorrs(floc_null,prctile_range = (1,99)):
    A = null_cancorrs_from_csv(floc_null)
    n_comps = A.shape[1]

    x = np.asarray(range(n_comps))
    null_yerr = np.zeros((2,n_comps))

    null_median = np.median(A,axis=0)
    null_bounds = np.percentile(A,prctile_range,axis=0)
    null_median = null_median

    null_median,null_yerr,extreme_flag = check_for_extremes(null_median,null_bounds)

    ls_null = 'dotted'
    label_txt = "null_dist (" + str(prctile_range[0]) + "%, " + str(prctile_range[1]) + "%)"

    fig,ax = plt.subplots()
    ax.errorbar(x,np.abs(null_median),yerr = null_yerr,linestyle = ls_null, label = label_txt, elinewidth = 1)

    return fig,ax,extreme_flag


def format_cancorrplot_axes(pairname,fig,ax,reg_val,prctile_range,num_sigcomps,extreme_vals = False):
    pairname = pairname.replace("_"," ")

    if reg_val == 0.:
        title_string = "Data vs. null canonical correlations (" + r"$n_{sig}=$" + str(num_sigcomps) + ")\n" + pairname + "\n" + "(no regularization; first 100 PCs)"
    elif np.isreal(reg_val):
        title_string = "Data vs. null canonical correlations (" + r"$n_{sig}=$" + str(num_sigcomps) + ")\n" + pairname + "\n" + "regularization val. = 1e" + str(int(np.log10(reg_val)))
    else:
        title_string = "Data vs. null canonical correlations\n" + pairname 
        
    xlabel = "correlation component"

    ax.set_title(title_string)
    ax.set_xlabel(xlabel)
    if extreme_vals:
        ax.set_ylabel(r"correlation value ($-\log_{10}(1-\rho)$)")
    else:
        ax.set_ylabel(r"correlation value ($\rho$)")
    return fig,ax

def null_cancorrs_from_csv(floc):
    with open(floc,'r') as fin:
        text_list = list(csv.reader(fin))

    cancorr_mtx = np.asarray([np.loadtxt(R) for R in text_list])
    return cancorr_mtx

def check_for_extremes(null_med,null_bounds,thresh=-6):
    max_med = max(np.max(np.log10(1-null_med)),-25)
    if max_med < thresh:
        null_med = - np.log10(1-null_med)
        null_med[null_med == np.inf] = 25.
        null_bounds = -np.log10(1-null_bounds)
        null_bounds[null_bounds == np.inf] = 25.
        extreme_flag = True
        #debug code:
        print("Extreme values found: max_med = 1e" + str(max_med))
    else:
        extreme_flag = False
        print("No extreme values found: max_dist = 1e" + str(max_med))

    null_yerr = np.abs(null_bounds - null_med)
    return null_med,null_yerr,extreme_flag

def find_sigcomps(floc_data,floc_null):
    null_dist = null_cancorrs_from_csv(floc_null)
    data_cancorrs = np.loadtxt(floc_data,delimiter=",")
    unc_pvals = np.sum(null_dist > data_cancorrs, axis=0) #+ 0.5/len(data_cancorrs)
    unc_pvals[unc_pvals > 1] = 1
    is_sig, fdr_pvals = fdrcorrection(unc_pvals, alpha=0.05)
    return is_sig,fdr_pvals


def plot_ndim_vs_reg(dim_vs_reg_list, regvals):
    n_regvals = len(regvals)
    n_pairnames = len(dim_vs_reg_list[0][0])
    n_dimvals = len(dim_vs_reg_list[0][1])

    fig,ax = plt.subplots()
    colors = plt.cm.rainbow(np.linspace(0,1,n_pairnames))
    for i in range(n_pairnames):
        pairname_i = dim_vs_reg_list[0][0][i]
        dimvals_i = [dim_vs_reg_list[k][1][i] for k in range(n_regvals)]

        ax.plot(np.log10(regvals), dimvals_i, 'o-', markersize=4, c=colors[i], linewidth=1, label=pairname_i.replace("_"," "))

    ax.set_title("Significance dimension vs. regularizer\n" + r" $(\alpha_{FDR} = 0.05)$")
    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel(r"number of significant dimensions $(\alpha=0.05)$")
    ax.legend()

    return fig, ax

if __name__=="__main__":
    main(sys.argv)
