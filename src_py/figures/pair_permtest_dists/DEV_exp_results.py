import re
import os
import glob
import json
import scipy
import argparse
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import fdrcorrection

# global variables 

def_fig_size = (24, 24)
def_label_fontsize = 7 

def_pattern='*X_*_dists'


# def_heatmap_vars = ["Wp_XY", "empirical_pval"]
def_heatmap_vars = ["Wp_XY", "empirical_pval"]
def_scatter_vars = ["Wp_XY", "Y_type"] 

# exp_outtype="All_vs_AllNull/X_ICA15_Amps_Psim_dists/ICA15_Amps_Psim_vs_Schaefer100_Amps_Psim_null-subjectPerms.json"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty paired-null distance distribution summaries
########################################################################################################################
def one_pair_plot(fpath, fig_title=None, verbose=True, debug=False):
    with open(fpath,'r') as fin:
        outputs = json.load(fin)
    data_diff = outputs[0]["Wp_XY"]
    null_diffs = pd.DataFrame(outputs[1:])

    # compute p-value from 2-sided test against empirical CDF (enforcing inf(p)=1/N)
    data_pval = 1 - np.mean(data_diff > null_diffs["Wp_XY"].to_numpy())
    data_pval = min(data_pval, 1 - data_pval)
    if data_pval < 1/len(null_diffs):
        data_pval = 1/len(null_diffs)

    g = sns.displot(data=null_diffs, x="Wp_XY", kind="hist", kde=True, hue="permtype")
    g.refline(x=data_diff, linestyle="--", color="red", label="data distance")

    if fig_title is None:
        X_type = outputs[0]["X_type"]
        Y_type = outputs[0]["Y_type"]
        fig_title = f"{X_type}_vs_{Y_type}\nreal vs. permuted p-Wasserstein distances"
    
    g.fig.suptitle(fig_title)

    if verbose:
        print(f"The approximate empircal p-value for data vs. null distance of {data_diff} is {data_pval}")

    return g, data_pval
########################################################################################################################

# heatmap plotting
########################################################################################################################
def _get_heatmap_inputs(alldata_grid, heatmap_vars=def_heatmap_vars, debug=False):
    xnamelist = [list(set(i[0]["X_type"]))[0] for i in alldata_grid]
    ynamelist = [list(set(j["Y_type"]))[0] for j in alldata_grid[0]]


    valuegrid = {}

    for varname in heatmap_vars:
        print(f"loading {varname}...")
        try:
            vals = np.squeeze(np.array([[j[varname].to_numpy() for j in i] for i in alldata_grid]))
            valuegrid[varname] = vals
            if debug:
                print(f"variable has grid of values: \n{vals}")
        except ValueError:
            new_entry = [[j[varname].to_numpy() for j in i] for i in alldata_grid]
            if debug:
                ### debugging code ###
                print(f"found data inmogeneity in {varname} readin. attempted new entry has data of following shapes and values:")
                print([var.shape for var in new_entry])
                print("corresponding to pairs:")
                print([[(j["X_type"],j["Y_type"]) for j in i] for i in alldata_grid])
                # print(new_entry)
                ### debugging code ###
            

    if debug:
        ### debugging code ###
        print(f"Names of {len(xnamelist)} 'X' spaces: \n{xnamelist}")
        print(f"Names of {len(ynamelist)} 'Y' spaces: \n{ynamelist}")
        print(f"Entries in list of grid values have the following shapes: \n{[valuegrid[var].shape for var in heatmap_vars]}")
        print("First entry in valuegrid: ", np.array(valuegrid[heatmap_vars[0]]))
        print(f"Generating one heatmap for each of the following set of variables: \n{heatmap_vars}")
        print("")
        ### debugging code ###
    return xnamelist, ynamelist, valuegrid


def generate_clustermaps(
        xnamelist,
        ynamelist,
        valuegrid,
        linkage_var = "Wp_XY",
        cluster_method = "average",
        fig_size = def_fig_size,
        label_fontsize = def_label_fontsize,
        outdir = None,
        write_mode = True
        ):
    clustervars = list(valuegrid.keys())

    assert linkage_var in clustervars, f"Value does not include variable \"{linkage_var}\", the specified common linkage operator"
    print(f"Using \"{linkage_var}\" as linkage variable while generating clustermaps")

    fig_dict = {}
    
    for varname in clustervars:
        fig_dict[varname] = plot_clustermap(
                xnamelist,
                ynamelist,
                valuegrid,
                cluster_method = cluster_method,
                linkage_var = linkage_var,
                display_var = varname,
                enf_sym = True,
                fig_size = fig_size,
                label_fontsize = label_fontsize,
                outdir = outdir,
                write_mode = write_mode
                )

        # can i turn list figure set into something that shows everything?



def plot_clustermap(
        xnamelist,
        ynamelist,
        valuegrid,
        cluster_method = "average",
        linkage_var = "Wp_XY",
        display_var = "empirical_pval",
        enf_sym = False,
        label_fontsize = def_label_fontsize,
        fig_size = def_fig_size,
        outdir = None,
        write_mode = True,
        debug = True
        ):

    print(f"enforcing symmetry in \'{linkage_var}\' values.")
    linkage_vals = _enforce_symmetry(valuegrid[linkage_var], fill_val=0)
    import scipy.cluster.hierarchy as hc
    xlinkage = hc.linkage(squareform(linkage_vals), method=cluster_method, optimal_ordering=True)
    if debug:
        print(f"found {np.count_nonzero(xlinkage < 0)} negative linkage values") 
        print(f"found {np.count_nonzero(np.isnan(xlinkage))} NaN linkage values")
        print(f"found {np.count_nonzero(np.isinf(xlinkage))} infinite linkage values")

    if enf_sym:
        print(f"enforcing symmetry in \'{display_var}\' values.")
        display_vals = _enforce_symmetry(valuegrid[display_var], fill_val=0)
        try:
            assert xnamelist == ynamelist
        except AssertionError:
            if debug:
                print(f"namelists are unequal in forced symmetric case! xnamelist: {len(xnamelist)} entries, ynamelist: {len(ynamelist)} entries")
                # print(f"namelists are unequal in forced symmetric case! \nxnamelist: {len(xnamelist)} entries\nynamelist: {len(ynamelist)} entries")
            ynamelist = xnamelist
    else:
        display_vals = valuegrid[display_var]

    assert linkage_vals.shape==display_vals.shape, "linkage and display values must have same dimensions!"
    
    print(f"Plotting grid of '{display_var}' values...")

    xticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in xnamelist]
    yticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in ynamelist]
    
    rb_title = f"Clustermap plot of {display_var}"

    if "pval" in display_var:
        display_vals = squareform(correct_pvals(triu_vals(display_vals,k=1)))
        np.fill_diagonal(display_vals, np.nan)
        display_vals = -np.log10(2*display_vals)
        rb_title = rb_title + " (-log10(2p))"
        np.nan_to_num(display_vals, nan=-1, copy=False)
    if "Wp_XY" in display_var:
        display_vals[ display_vals==0 ] = np.nan
        display_vals = np.log10(display_vals)
        rb_title = rb_title + " (log10(W_p))"
        max_num = -1.1*np.nanmax(np.abs(display_vals))
        if debug:
            print(f"replacing NaNs in log10(Wp_XY) with {max_num}")
        np.nan_to_num(display_vals, nan=max_num, copy=False)

    if np.count_nonzero(np.isnan(display_vals)) > 0:
        if debug:
            print(f"{np.count_nonzero(np.isnan(display_vals))} NaNs removed removed from \'display_vals\' for var \"{display_var}\"")
        np.nan_to_num(display_vals, nan=-1, copy=False)
        

#   if debug:
#       ### debugging code ###
#       print(f"xticklabels: {xticklabels[0]}")
#       print(f"yticklabels: {yticklabels[0]}")

    from compare_topostats import _plot_clustermap

    g = _plot_clustermap(
        display_vals, 
        cluster=True,
        cluster_method=cluster_method,
        cm_title = rb_title,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        xlinkage=xlinkage,
        ylinkage=xlinkage,
        cmap = sns.color_palette("Spectral", as_cmap=True),
        fig_size=fig_size,
        write_mode=False,
        debug=debug
        )

    fig = g.fig
    ax = g.ax_heatmap
    ax.set(title = rb_title)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    if write_mode:
        outpath = os.path.join(outdir, f"clustermap_{display_var}.png").replace(" ","")
        _write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()

    return g

# heatmap plot utilities
########################################################################################################################
########################################################################################################################


# compute secondary statistics
########################################################################################################################
# add an empirical p-val to a dataframe containing only a single data-derived distance and its null counterparts
def _add_emp_pval(df, check_match=True):
    datarow = df[df["datatype"] == "Data"]
    nullrows = df[df["datatype"] == "Null"]

    Wp_XY = datarow["Wp_XY"].to_numpy()
    Wp_XYnull = nullrows["Wp_XY"].to_numpy()

    if len(Wp_XYnull) == 0:
        empirical_pval = np.nan
    else:
        if check_match:
            check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
            err_str =  f"data row and null rows are not of matching type: \n{[[col, set(datarow[col]), set(nullrows[col])] for col in check_cols]}"
            assert all( [ set(datarow[col]) == set(nullrows[col]) for col in check_cols ] ), err_str

        prop_lower = np.mean(Wp_XY > Wp_XYnull)
        # compute p-value from 2-sided test against empirical CDF (enforcing inf(p)=1/N)
        # empirical_pval = max(1/len(Wp_XYnull), 2*min(prop_lower, 1 - prop_lower))
        # compute p-value from 1-sided test against empirical CDF (enforcing inf(p)=1/N)
        # empirical_pval = max(1/len(Wp_XYnull), 1 - prop_lower)
        # compute p-value from 1-sided test against empirical CDF (enforcing inf(p)=1/N)
        empirical_pval = max(1/len(Wp_XYnull), prop_lower)

    df["empirical_pval"] = [empirical_pval] + [np.nan]*len(Wp_XYnull)

    return df

def correct_pvals(pval_vec):
    rmv_idx = np.isnan(pval_vec) + (pval_vec < 0)
    _, corr_pvals = fdrcorrection(pval_vec[ rmv_idx == False ])
    pval_vec[ rmv_idx == False ] = corr_pvals
    return pval_vec
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def pull_data(
        parent_dir, dir_pattern='X_*_dists', f_pattern = '*_vs_*', 
        name_type="exp_results", check_pval=True, data_only=True, debug=True
        ):
    dirlist = glob.glob(os.path.join(parent_dir, dir_pattern))
    dirlist.sort()

    fpath_grid = [ glob.glob(os.path.join(X_dir, f"{f_pattern}.json")) for X_dir in dirlist ]
    [i.sort() for i in fpath_grid]

    if data_only:
        print("Only retaining information from datatype \"Data\" (discarding \"Null\"-type data after necessary computations)")
    
    alldata_grid = [ [ _load(
        fpath, data_only=data_only, name_type=name_type, check_pval=check_pval
        ) for fpath in X_sublist ] for X_sublist in fpath_grid ]

    if debug:
        ### debugging code ###
        print(f"Pulling from fpath_grid w/ 00 entry: \n{fpath_grid[0][0]}")
        if not isinstance(alldata_grid[0], list):
            print(f"alldata_grid loadin variable is not nested lists, but instead has following structure: \n{[type(x) for x in alldata_grid]}")
        try:
            samp = alldata_grid[0][0]
            print(f"00 entry of alldata_grid: \n{samp}")
        except IndexError:
            print(f"0-row entry of alldata_grid: \n{alldata_grid[0]}")
        if isinstance(samp, str):
            gridlist_shape = [len(alldata_grid), np.mean([len(i) for i in alldata_grid]), np.mean([len(i) for j in alldata_grid for i in j])]
        else:
            gridlist_shape = [len(alldata_grid), np.mean([len(i) for i in alldata_grid]), set([i.shape for j in alldata_grid for i in j])]

        print(f"gridlist has \"shape\" given by \n{gridlist_shape}")
        ### debugging code ###

    return alldata_grid


def _load(input_fpath, name_type="exp_results", check_pval=True, data_only=True, parse_longname=False):
    with open(input_fpath, 'r') as fin:
        data_df = pd.DataFrame(json.load(fin))

    if parse_longname:
        data_df[["X_mod","X_feat","X_diff"]] = data_df["X_type"].str.split('_', n=2, expand=True)
        data_df[["Y_mod","Y_feat","Y_diff"]] = data_df["Y_type"].str.split('_', n=2, expand=True)
        data_df.drop(["X_type","Y_type"], axis=1, inplace=True)


    if check_pval:
        if "empirical_pval" not in data_df.columns:
            data_df = _add_emp_pval(data_df)

    if data_only:
        data_df = data_df[data_df["datatype"] == "Data"]

    return data_df


# Enforces symmetry under assumption 'gridlist' produced by a pairwise process skipping its first trivial pairing
def _enforce_symmetry(mtx, debug=False, fill_val=np.nan):
    assert len(mtx.shape)==2, "Only valid for matrix inputs"
    assert (mtx.shape[0]-1)==mtx.shape[1], f"Input matrix assumed to have shape (n,n-1): instead, given matrix has shape {mtx.shape}"

    # takes values from upper diagonal
    sym_mtx = squareform(triu_vals(mtx, k=0))
    np.fill_diagonal(sym_mtx, fill_val)

    assert np.allclose(sym_mtx, sym_mtx.T, equal_nan=True), f"Symmetrization failed: \"sym_mtx\" is \n{sym_mtx}"

    return sym_mtx


def triu_vals(A, k=1):
    n = min(A.shape)
    vals = A[np.triu_indices(n, k)]
    return vals


def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):
    fig.set_size_inches(fig_size, forward=False)
    fig.savefig(outpath, dpi=600)
    print(f"saved to {outpath}")
########################################################################################################################



########################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Create and write summary figures summarizing bootstrapped distance data"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="",
        help="directory with name of type []_vs_[] containing bootstrapped distance outputs"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="figure output directory"
    )
    parser.add_argument(
        "-n",
        "--name_type",
        type=str,
        default="exp_results",
        help="Specifies the class of naming conventions used for the target data"
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default=def_pattern,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-H",
        "--do_heatmap",
        default=False,
        action="store_true",
        help="Generate heatmaps of pairwise summary comparisons over varying parameters in each pair (CURRENTLY UNUSED)"
    )
    parser.add_argument(
        "-S",
        "--do_scatter",
        default=False,
        action="store_true",
        help="Generate scatterplots of per-space stability summary quantities"
    )
    parser.add_argument(
        "-w",
        "--write_mode",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)

    debug=True

    alldata_grid = pull_data(
            args.input_dir, 
            dir_pattern=args.pattern,
            data_only = True,
            check_pval = True,
            name_type=args.name_type
            )
    if debug:
        intm_dir = os.path.join(args.output_dir, "alldata_grid")
        if not os.path.isdir(intm_dir):
            os.mkdir(intm_dir)
        for i, sublist in enumerate(alldata_grid):
            for j, df in enumerate(sublist):
                fname = f"alldata_col{i}_row{j}.csv"
                df.to_csv(os.path.join(intm_dir, fname))

    xnamelist, ynamelist, valuegrid = _get_heatmap_inputs(alldata_grid, heatmap_vars=def_heatmap_vars)

    if debug:
        for name in def_heatmap_vars:
            savepath = os.path.join(args.output_dir, f"{name}.csv")
            np.savetxt(savepath, valuegrid[name])
            print(f"wrote value grid for value \"{name}\" to \"{savepath}\"")

    generate_clustermaps(
            xnamelist, 
            ynamelist, 
            valuegrid, 
            linkage_var = "Wp_XY",
            cluster_method = "average",
            label_fontsize = def_label_fontsize,
            outdir=args.output_dir,
            write_mode=args.write_mode
            )

    if args.do_scatter:
        scatter_df, hue_var, style_var = _get_scatter_df(alldata_grid, scatter_vars=def_scatter_vars, name_type=name_type)
        generate_scatter_plots(
                args.output_dir, 
                scatter_df, 
                hue_var=hue_var, 
                style_var=style_var, 
                write_mode=args.write_mode
                )

