import os
import ot                           # optimal transport library
import ast
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from PD_weighted_Wasserstein import weighted_Wasserstein_loss as Wp_wt


def summarize_topostats(prevpath_list_fpath, outdir):

    prevpath_list, B1path_list, barsXpaths_list = _get_pathlists(prevpath_list_fpath)
    list_fpath_nametype = os.path.basename(prevpath_list_fpath).split('.')[0]

    prevs_list = [ np.genfromtxt(fpath) for fpath in prevpath_list ]
    B1matches_list = [ np.genfromtxt(fpath) for fpath in B1path_list ]
    bars_list = [ _get_bars(fpath) for fpath in barsXpaths_list ]
    name_list = [ _get_name(fpath) for fpath in barsXpaths_list ]

    uw_title = "Wasserstein Dist. from\n" + "H1 persistence diagrams"
    savepath = os.path.join(outdir,f"PD_Wpclustermap_{list_fpath_nametype}.png")
    make_Wp_clustermap(bars_list, name_list, cm_title=uw_title, write_mode=True, savepath=savepath)
    
    prev_title = "Wasserstein Dist. from\n" + "H1 persistence diagrams\n" + "(prevalence-weighted)"
    savepath = os.path.join(outdir,f"prevwt_PD_Wpclustermap_{list_fpath_nametype}.png")
    make_Wp_clustermap(bars_list, name_list, weights_list=prevs_list, cm_title=prev_title, write_mode=True, savepath=savepath)

#    B1match_title = "Wasserstein Dist. from\n" + "Matched Cycles per Bootstrap"
#    savepath = os.path.join(outdir,f"B1match_Wpclustermap_{list_fpath_nametype}.png")
#    B1_Wp = make_Wp_clustermap(B1matches_list, name_list, cm_title=B1match_title, write_mode=True, savepath=savepath)
# 
#    B1match_pvals_title = "Significant Differences in\n" + "Matched Cycles per Bootstrap\n" + "log10(p) from T-test"
#    savepath = os.path.join(outdir,f"B1match_pval_heatmap_{list_fpath_nametype}.png")
#    B1_pvals = make_pvals_heatmap(B1matches_list, name_list, cm_title=B1match_pvals_title, write_mode=True, savepath=savepath)
#    np.savetxt(os.path.join(outdir,f"B1match_pvals_{list_fpath_nametype}.txt"), B1_pvals)
# 
#    plt.scatter( np.log10(np.abs(B1_pvals.flatten())), B1_Wp.flatten())
#    plt.title("Wasserstein distance vs. T-test log10(p-value)")
#    plt.xlabel("log10(p)")
#    plt.ylabel("Wasserstein distance")
#    savepath = os.path.join(outdir,f"B1match_Wp_vs_pval_{list_fpath_nametype}.png")
#    plt.savefig(savepath, dpi=600)

    prevalence_vs_persistence(
            prevs_list, 
            bars_list,
            title = "Homology Persistence vs. Homology Prevalence",
            x_label = "Prevalence Score",
            # y_label = "Persistence (death/birth)",
            y_label = "Persistence (death - birth)",
            # savepath= os.path.join(outdir,f"ratio_persistence_vs_prev_{list_fpath_nametype}.png")
            savepath= os.path.join(outdir,f"persistence_vs_prev_{list_fpath_nametype}.png")
            )


def make_Wp_clustermap(distribution_list, name_list, weights_list=None, 
        cm_title="Wasserstein Distances", write_mode=True, savepath="Wp_clustermap.png"):
    N = len(distribution_list)

    Wp_clustermap = np.zeros((N,N))

    for i, dist_i in enumerate(distribution_list):
        for j in range(i+1,N):
            dist_j = distribution_list[j]
            
            try:
                if weights_list:
                    Wp_clustermap[i,j] = comp_Wp_dist(dist_i, dist_j, w1=weights_list[i], w2=weights_list[j])
                else:
                    Wp_clustermap[i,j] = comp_Wp_dist(dist_i, dist_j)
            except Exception as err:
                Wp_clustermap[i,j] = np.nan
                print(f"comparison between {name_list[i]} and {name_list[j]} failed.")
                print(f"    error msg: {err}")
                if not weights_list is None:
                    print(type(weights_list[i]), type(weights_list[j]))

    # Wp_clustermap = Wp_clustermap + Wp_clustermap.T
    np.savetxt(savepath.replace('.png','.txt'), Wp_clustermap)
    _plot_clustermap(
            Wp_clustermap, symmetrize=True,
            cm_title=cm_title, name_list=name_list, write_mode=write_mode, savepath=savepath)
    return Wp_clustermap


def make_pvals_heatmap(distribution_list, name_list, pvals_type="welch_T",
        cm_title="p-Values", write_mode=True, savepath="pval_clustermap.png"):
    prob_type = _specify_prob(pvals_type)
    N = len(distribution_list)

    pval_clustermap = np.zeros((N,N))
    bonferroni = N^2 - N

    for i, dist_i in enumerate(distribution_list):
        for j, dist_j in enumerate(distribution_list):
            dist_j = distribution_list[j]
            pval_clustermap[i,j] = prob_type(dist_i, dist_j)*bonferroni + 1e-323

    np.fill_diagonal(pval_clustermap, 1)

    _plot_clustermap(np.log10(np.abs(pval_clustermap)), cm_title=cm_title, name_list=name_list, write_mode=write_mode, cluster=False, savepath=savepath)
    return pval_clustermap


def _specify_prob(method='welch_T'):
    prob_methods = {
            'welch_T': _t_test,
            'KS_test': stats.ks_2samp
            }
    prob_type = prob_methods.get(method, lambda arg: print(f"Unknown probability test type: {arg}"))
    return prob_type

def _t_test(dist1, dist2, student=False, permutations=None):
    tstat, pval = stats.ttest_ind(dist1, dist2, 
            equal_var=student, permutations=permutations, nan_policy="raise")
    return pval


def _plot_clustermap(values, symmetrize=False, cm_title="Heatmap", name_list=None, write_mode=True, savepath="", cluster=True):
    fig_sz = (12, 12)

    if symmetrize:
        values = (values + values.T)/2
    if cluster:
        import scipy.cluster.hierarchy as hc
        import scipy.spatial as sp
        linkage = hc.linkage(sp.distance.squareform(values), method="ward", optimal_ordering=True)

        kws = dict(cbar_kws=dict(orientation='vertical'), figsize=fig_sz)
        dgram_ratio = 0.1618

        g = sns.clustermap(values, row_linkage=linkage, col_linkage=linkage, cmap = "Blues",
                xticklabels=name_list, yticklabels=name_list, dendrogram_ratio=dgram_ratio, **kws)
        g.fig.suptitle(cm_title, fontsize="xx-large")
        g.fig.set_size_inches(fig_sz, forward=False)
        g.ax_cbar.set_position([g.cbar_pos[0], 1-dgram_ratio/10, dgram_ratio/10, g.ax_row_dendrogram.get_position().width])
        g.ax_cbar.tick_params(labelrotation=0)
        g.ax_col_dendrogram.set_visible(False)          #suppress column dendrogram
    else:
        g, ax = plt.subplots()
        sns.heatmap(values, square = True, cbar = True, ax=ax, cmap = "Blues",
                xticklabels=name_list, yticklabels=name_list)
        ax.xaxis.tick_top()
        plt.title(cm_title)
        plt.tight_layout()  # neccessary to get the x-axis labels to fit
        g.set_size_inches(fig_sz, forward=False)

    plt.xticks(fontsize=6,rotation=90)
    plt.yticks(fontsize=6) #rotate the tick labels
    if write_mode:
        g.savefig(savepath, dpi=600)
        print(f"saved to {savepath}")
        plt.close()
    else:
        return fig

# computes (potentially sliced) Wasserstein distance between two distributions
### dist1 and dist2 are instances of np.array
### w1 and w2 should be 1d vectors with len(wN)=distN.shape[0]
def comp_Wp_dist(dist1, dist2, w1=None, w2=None, n_proj=500, p=2):

    if len(dist1.shape) > 1:     # assumes dist1 and dist2 are instances of np.array
        dim1, dim2 = dist1.shape[1], dist2.shape[1]
        assert np.array_equal([dim1, dim2], [2, 2]), "This computation of Wasserstein distance assumes persistence diagrams (in R2) as input."
    else:
        dim1 = 1

    if not w1 is None:
        if w1.size == 1:
            w1 = np.array([w1])
    if not w2 is None:
        if w2.size == 1:
            w2 = np.array([w2])

    try:
        if dim1 > 1:
            Wp_dist = Wp_wt(
                    dist1, dist2, 
                    w1=w1, w2=w2, p=p,
                    wtfn_type="diff", ot_imp="POT"
                    )
        else:
            # does not implement weighting for the 1-dimensional case
            Wp_dist = np.power(ot.wasserstein_1d(dist1, dist2, p=p), 1/p)
    except Exception as err:
        print(f"         Wp_dist computation failed: {err}")
        if not w1 is None:
            if w1.size==1:
                print(f"dist1: {dist1}")
                print(f"w1: {w1}")
        if not w2 is None:
            if w2.size==1:
                print(f"dist2: {dist2}")
                print(f"w2: {w2}")

    return Wp_dist


def prevalence_vs_persistence(prevs_list, bars_list, 
        title="", x_label="", y_label="", savepath=""):
    prevalence_scores = np.concatenate([prev.flatten() for prev in prevs_list])
    # persistence_list = [np.exp(np.diff(np.log(bars))) for bars in bars_list]
    persistence_list = [np.diff(bars) for bars in bars_list]
    persistence = np.concatenate([pers.flatten() for pers in persistence_list])

    x_exts = [min(prevalence_scores), max(prevalence_scores)]
    xlims = [x_exts[0] - 0.05*np.diff(x_exts), x_exts[1] + 0.05*np.diff(x_exts)]
    y_exts = [min(persistence), max(persistence)]
    ylims = [y_exts[0] - 0.05*np.diff(y_exts), y_exts[1] + 0.05*np.diff(y_exts)]

    from scipy.stats import gaussian_kde
    xy = np.vstack([prevalence_scores, persistence])
    dens = gaussian_kde(xy)(xy)

    fig = plt.figure()
    plt.scatter(x=prevalence_scores, y=persistence, c=dens, s=4)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    fig.savefig(savepath, dpi=600)
    print(f"Prevalence vs. Persistence plot saved to {savepath}")


def _get_pathlists(prevpath_list_fpath):
    with open(prevpath_list_fpath, 'r') as fin:
        topostats_path_list = fin.read().split()

    prevpath_list = [ fpath for fpath in topostats_path_list if "prevalence" in fpath ]
    B1path_list = [ fpath for fpath in topostats_path_list if "B1match" in fpath ]
    prevbase = os.path.basename(prevpath_list[0])
    barsXpaths_list = [ fpath.replace(prevbase, "bars_X.txt") for fpath in prevpath_list ]

    ### debug code ###
    # print(f"List of paths to prevalence scores: {prevpath_list[0:1]}...")
    # print(f"Prevalence score (generic) filename: {prevbase}")
    # print(f"List of paths to (full) persistence barcodes: {barsXpaths_list[0:1]}...")
    # print("")
    ### debug code ###
    return prevpath_list, B1path_list, barsXpaths_list

def _get_bars(bars_fname, homdim=1):
    with open(bars_fname, 'r') as fin:
        all_bars = ast.literal_eval(fin.read())
        bars = all_bars[homdim]  # "bars" dictionary is indexed by homology dimension
    return np.array(bars)

def _get_name(fpath):
    dname = os.path.basename(os.path.dirname(fpath))
    name = dname.replace('phom_data_','').replace('_dists','').replace('_ztrans','-ztrans').replace('_',' ')
    return name


################################################################################################################
# parses input, streams output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-p",
        "--prevpath_list_fpath",
        type=str,
        default="",
        help="filepath containing list of filepaths pointing to distributions"
    )
    parser.add_argument(
        "-w",
        "--write",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    parser.add_argument(
        "-f", "--fig_outdir", 
        default=None, 
        type=str, 
        help="output filepath to persistence diagram image"
    )
    args = parser.parse_args()
    summarize_topostats(args.prevpath_list_fpath, args.fig_outdir)
