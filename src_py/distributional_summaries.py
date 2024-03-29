import os
import re
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _get_pathlist(pathlist_fpath):
    with open(pathlist_fpath,'r') as fin:
        pathlist = fin.read().split()
    return pathlist

#######################################################################################################################
def all_swarmplots(pathlist, stat_types=["prevalence", "B1match"], write_mode=True):
    for stat_type in stat_types:
        reduced_pathlist = [path for path in pathlist if stat_type in path]
        color_swarmplots(reduced_pathlist, stat_type, write_mode=write_mode)


def color_swarmplots(pathlist, stat_type="prevalence", write_mode=True):
    stat_df = _pathlist_to_df(pathlist)
    stat_df.rename(columns={"statvals": stat_type}, inplace=True)
    
    ### debug code ###
    print(stat_df)
    ### debug code ###

    # fig1 = _single_swarmplot(stat_df, stat_type, x="log10(rank)", hue="metric", write_mode=write_mode)
    # fig2 = _single_swarmplot(stat_df, stat_type, x="log10(rank)", hue="method", write_mode=write_mode)
    # fig3 = _single_swarmplot(stat_df, stat_type, x="method", hue="feature", write_mode=write_mode)
    # fig4 = _single_swarmplot(stat_df, stat_type, x="feature", hue="metric", write_mode=write_mode)


    #_do_catplot(stat_df, stat_type, x="log10(rank)", hue="method", coltype="metric", rowtype="feature")
    _do_catplot(stat_df, stat_type, x="log10(rank)", hue="method", coltype="metric")
    _do_catplot(stat_df, stat_type, x="log10(feat_num)", hue="method", coltype="metric")
    _do_catplot(stat_df, stat_type, x="method", hue="metric", coltype="feature")

    if not write_mode:
        plt.show()


def _single_swarmplot(dataframe, stat_type, x=None, hue=None, 
        write_mode=True, outdir=os.getcwd()):

    print(f"swarmplot of {stat_type} values with categorical variable {x}, color-coded by {hue}...")
    
    fig, ax = plt.subplots()
    if x=="log10(rank)":
        sns.swarmplot(data=dataframe, x=x, y=stat_type, hue=hue, dodge=True, legend=True, ax=ax, native_scale=True, size=1)
    else:
        sns.swarmplot(data=dataframe, x=x, y=stat_type, hue=hue, dodge=True, legend=True, ax=ax, size=1)

    if stat_type=="prevalence":
        ax.set_ylim(-0.05,1.05)

    plt.title(f"Summary of {stat_type} distributions: \n {x} and {hue}")
    plt.xlabel(x)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6,rotation=90) #rotate the tick labels
    plt.axis('equal') #make it square
    plt.tight_layout()  # neccessary to get the x-axis labels to fit
    if write_mode:
        savepath = os.path.join(outdir, "swarmplots", f"swarmplot_{stat_type}_{x}_{hue}.png")
        fig.set_size_inches((8,8), forward=False)
        fig.savefig(savepath, dpi=600)
        print(f"saved to {savepath}")

    return fig

def _do_catplot(dataframe, stat_type, x=None, hue=None, coltype=None, rowtype=None,
        write_mode=True, outdir=os.getcwd()):

    print(f"swarmplot facets of {stat_type} values with categorical variable {x} and color-coded by {hue}")
    print(f"facet column: {coltype}")
    print(f"facet row: {rowtype}")

    if "log10" in x:
        scale=True
        dodge=True
    else:
        scale=False
        dodge=False

    if rowtype:
        fg = sns.catplot(
                data = dataframe, x=x, y=stat_type, hue=hue,
                kind="swarm", size=1, dodge=dodge,
                row=rowtype, col=coltype, native_scale=scale, legend=True,
                )
    else:
        fg = sns.catplot(
                data = dataframe, x=x, y=stat_type, hue=hue,
                kind="swarm", size=1, dodge=dodge,
                col=coltype, col_wrap=2, native_scale=scale, legend=True,
                )

    if stat_type=="prevalence":
        fg.set(ylim = (-0.05,1.05))

    fg.set_axis_labels(x_var=x, y_var=stat_type)
    fg.set_titles("{col_name} {col_var}")
    fg.tight_layout()
    

    if write_mode:
        savepath = os.path.join(outdir, "swarmplots", f"catplot_{stat_type}_{x}_{hue}_{coltype}-{rowtype}.png")
        fg.height = 16
        fg.aspect = 1
        fg.savefig(savepath, dpi=600)
        print(f"saved to {savepath}")

    return fg


def _pathlist_to_df(pathlist):
    namelist = [os.path.basename(os.path.dirname(fpath)).replace('phom_data_','') for fpath in pathlist]
    labels = [name.split('_') for name in namelist]
    methods, ranks = _pull_methods_ranks([label[0] for label in labels])
    features = [label[1] for label in labels]
    feat_nums = [_pull_feat_num(ranks[i], feat) for i, feat in enumerate(features)]
    metrics = ['-'.join(label[2:]).replace('-dists','') for label in labels]
    distribution_list = [np.genfromtxt(fpath) for fpath in pathlist]

    statvals, uf_methods    = _unfold(distribution_list, methods)
    statvals, uf_ranks      = _unfold(distribution_list, ranks)
    statvals, uf_featnums   = _unfold(distribution_list, feat_nums)
    statvals, uf_features   = _unfold(distribution_list, features)
    statvals, uf_metrics    = _unfold(distribution_list, metrics)

    col_names = ["statvals", "method", "log10(rank)", "log10(feat_num)", "feature", "metric"]
    input_lists = [statvals, uf_methods, uf_ranks, uf_featnums, uf_features, uf_metrics]
    df = pd.DataFrame(index=col_names, data=input_lists).T

    return df

def _pull_methods_ranks(long_methods):
    split_methods_ranks = [_pull_rank(method) for method in long_methods]
    methods = [method[0] for method in split_methods_ranks]
    ranks = [method[1] for method in split_methods_ranks]
    return methods, ranks

def _pull_rank(long_method):
    if 'PROFUMO' in long_method:
        rank=33
        method="PROFUMO"
    elif 'Glasser' in long_method:
        rank=360
        method="Glasser"
    else:
        rank_pattern = re.compile('\d{1,4}')
        rank = re.search(r'\d{1,4}', long_method).group()
        method = long_method.replace(rank,'')
    return method, np.log10(float(rank))

def _pull_feat_num(rank, feature):
    rank = int(10**rank)
    if 'NM' in feature:
        feat_num = rank * (rank - 1) / 2
    elif 'Map' in feature:
        feat_num = rank * 91282
    elif 'Amps' in feature:
        feat_num = rank
    else:
        raise Exception("Unrecognized feature type")
    return np.log10(float(int(feat_num)))


def _unfold(distribution_list, metadata):
    statvals = [val for dist in distribution_list if dist.size > 1 for val in dist] + [np.float64(dist)
            for dist in distribution_list if dist.size == 1]
    uf_metadata = [metadata[idx] for idx, dist in enumerate(distribution_list) if dist.size > 1 for val in dist] + [metadata[idx] 
            for idx, dist in enumerate(distribution_list) if dist.size == 1]
    return statvals, uf_metadata
#######################################################################################################################


#######################################################################################################################
def show_all_histograms(pathlist):
    prev_scores = [np.array(np.genfromtxt(fpath)) for fpath in pathlist if "prevalence" in fpath]
    B1match_num = [np.array(np.genfromtxt(fpath)) for fpath in pathlist if "B1match" in fpath]
    namelist = [os.path.basename(os.path.dirname(fpath)).replace('phom_data_','') for fpath in pathlist if "prevalence" in fpath]

    nrows, ncols = _factor_pair(len(namelist))

    plot_hist_group(prev_scores, namelist, figshape=[nrows, ncols], title="Prevalence Scores")
    plot_hist_group(B1match_num, namelist, figshape=[nrows, ncols], title="Matched B1 Numbers")
    plt.show()


def plot_hist_group(values_list, names_list, figshape=None, title=''):

    fig, axes = plt.subplots(nrows=int(figshape[0]), ncols=int(figshape[1]))
    axes = axes.ravel()

    for idx, ax in enumerate(axes):
        distvals = values_list[idx]
        n_vals = distvals.size

        if n_vals < 2:
            distvals = [distvals]

        ax.hist(distvals, density=True)
        ax.set_title(names_list[idx])
        ax.set_ylabel(f"{n_vals} total counts")

    fig.suptitle(title)


def _factor_pair(N):
    if _check(N, np.sqrt(N)):
        nrows=np.sqrt(N)
        ncols=np.sqrt(N)
    else:
        pos_fac=int(np.sqrt(N))
        while not _check(N, pos_fac):
            pos_fac -= 1
        nrows=pos_fac
        ncols=N/pos_fac
    return nrows, ncols

def _check(N, pos_fac):
    div = N/pos_fac
    factor=( div == int(div) )
    return factor
#######################################################################################################################


################################################################################################################
# parses input, streams output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-f",
        "--pathlist_fpath",
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
        "-a",
        "--all_histograms",
        default=False,
        action="store_true",
        help="generate subplot array of histograms"
    )
    parser.add_argument(
        "-s",
        "--swarmplot",
        default=False,
        action="store_true",
        help="generate swarm plots"
    )
    args = parser.parse_args()

    # pathlist = _get_pathlist(args.pathlist_fpath)[:7]
    pathlist = _get_pathlist(args.pathlist_fpath)

    if args.all_histograms:
        show_all_histograms(pathlist)

    if args.swarmplot:
        all_swarmplots(pathlist, write_mode=args.write)
