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


# def_clustermap_vars = ["Wp_XY", "empirical_pval"]
def_clustermap_vars = ["Wp_XY", "empirical_pval", "Wp_XYNull_mean", "Wp_XYNull_std"]
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

    # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
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
def _get_heatmap_inputs(alldata_grid, clustermap_vars=def_clustermap_vars, debug=False):
    xnamelist = [list(set(i[0]["X_type"]))[0] for i in alldata_grid]
    ynamelist = [list(set(j["Y_type"]))[0] for j in alldata_grid[0]]


    valuegrid = {}

    for varname in clustermap_vars:
        try:
            vals = np.squeeze(np.array([[j[varname].to_numpy() for j in i] for i in alldata_grid]))
            if debug:
                print(f"variable has grid of values with shape: \n{vals.shape}")
        except ValueError:
            new_entry = [[j[varname].to_numpy() for j in i] for i in alldata_grid]
            if debug:
                ### debugging code ###
                print(f"found data inhomogeneity in {varname} readin. attempted new entry has data of following shapes and values:")
                print([var.shape for var in new_entry])
                print("corresponding to pairs:")
                print([[(j["X_type"],j["Y_type"]) for j in i] for i in alldata_grid])
                # print(new_entry)
                ### debugging code ###
        if 'pval' in varname:
            pval_type = alldata_grid[0][0]["pval_type"].values[0]
            if debug:
                print(f"p-value of type {pval_type} means varname becomes {varname}")
            if pval_type == "all":
                valuegrid[f"{varname}_right"] = vals
                valuegrid[f"{varname}_left"] = 1-vals
                vals2t =  2*np.min(np.stack((vals, 1-vals), axis=2), axis=2)
                valuegrid[f"{varname}_two-tailed"] = vals2t
                if debug:
                    print(f"2-sided p-values: \n{vals2t}")
                    exit()
                continue
            else:
                varname = f"{varname}_{pval_type}"

        valuegrid[varname] = vals
        print(f"\'{varname}\' gridded.")

    if debug:
        ### debugging code ###
        print(f"Names of {len(xnamelist)} 'X' spaces: \n{xnamelist}")
        print(f"Names of {len(ynamelist)} 'Y' spaces: \n{ynamelist}")
        print(f"Entries in list of grid values have the following shapes: \n{[valuegrid[var].shape for var in list(valuegrid.keys())]}")
        # print("First entry in valuegrid: ", np.array(valuegrid[clustermap_vars[0]]))
        print(f"Generating one heatmap for each of the following set of variables: \n{list(valuegrid.keys())}")
        print("")
        ### debugging code ###
    return xnamelist, ynamelist, valuegrid


def generate_clustermaps(
        xnamelist,
        ynamelist,
        valuegrid,
        onelink = False,
        linkage_var = "Wp_XY",
        cluster_method = "average",
        log_scale = True,
        fig_size = def_fig_size,
        label_fontsize = def_label_fontsize,
        outdir = None,
        write_mode = True
        ):
    dispvars = list(valuegrid.keys())

    if onelink:
        assert linkage_var in dispvars, f"Value does not include variable \"{linkage_var}\", the specified common linkage operator"
        print(f"Using \"{linkage_var}\" as linkage variable while generating clustermaps")
        linkvars = [linkage_var]
    else:
        print(f"Plotting clustermaps for all (linkage_val, display_var) value pairs (including self-pairs) in {dispvars}")
        linkvars = dispvars

    fig_dict = {}

    for linkage_var in linkvars:
        for display_var in dispvars:
            fig_dict[display_var] = plot_clustermap(
                    xnamelist,
                    ynamelist,
                    valuegrid,
                    cluster_method = cluster_method,
                    linkage_var = linkage_var,
                    display_var = display_var,
                    enf_sym = True,
                    log_scale = log_scale,
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
        log_scale = True,
        label_fontsize = def_label_fontsize,
        fig_size = def_fig_size,
        outdir = None,
        write_mode = True,
        debug = False
        ):

    print(f"enforcing symmetry in \'{linkage_var}\' linkage values.")
    linkage_vals = _enforce_symmetry(valuegrid[linkage_var], fill_val=0)
    import scipy.cluster.hierarchy as hc
    xlinkage = hc.linkage(squareform(linkage_vals), method=cluster_method, optimal_ordering=True)
    if debug:
        print(f"found {np.count_nonzero(xlinkage < 0)} negative linkage values") 
        print(f"found {np.count_nonzero(np.isnan(xlinkage))} NaN linkage values")
        print(f"found {np.count_nonzero(np.isinf(xlinkage))} infinite linkage values")

    if enf_sym:
        print(f"enforcing symmetry in \'{display_var}\' display values.")
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
    
    cm_title = f"Clustermap plot of {display_var} \n(clustered on {linkage_var})"

    if log_scale:
        if "pval" in display_var:
            display_vals = squareform(correct_pvals(triu_vals(display_vals,k=1)))
            np.fill_diagonal(display_vals, np.nan)
            display_vals = -np.log10(2*display_vals)
            cm_title = cm_title + " (-log10(2p))"
            np.nan_to_num(display_vals, nan=-1, copy=False)
        if "Wp_XY" in display_var:
            display_vals[ display_vals==0 ] = np.nan
            display_vals = np.log10(display_vals)
            cm_title = cm_title + " (log10(W_p))"
            max_num = -1.1*np.nanmax(np.abs(display_vals))
            print(f"replacing NaNs in log10({display_var}) with {max_num}")
            np.nan_to_num(display_vals, nan=max_num, copy=False)
        display_var = f"log-{display_var}"

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
        cm_title = cm_title,
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
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    if write_mode:
        outpath = os.path.join(outdir, f"cluster-on-{linkage_var}_of-{display_var}.png").replace(" ","")
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
def _add_emp_pval(df, check_match=True, pval_type="all", debug=False):
    datarow = df[df["datatype"] == "Data"]
    nullrows = df[df["datatype"] == "Null"]

    Wp_XY = datarow["Wp_XY"].to_numpy()
    Wp_XYnull = nullrows["Wp_XY"].to_numpy()

    if debug:
        print(f"Adding p-value of type {pval_type}")

    if len(Wp_XYnull) == 0:
        empirical_pval = np.nan
    else:
        if check_match:
            check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
            err_str =  f"data row and null rows are not of matching type: \n{[[col, set(datarow[col]), set(nullrows[col])] for col in check_cols]}"
            assert all( [ set(datarow[col]) == set(nullrows[col]) for col in check_cols ] ), err_str

        prop_lower = np.mean(Wp_XY > Wp_XYnull)
        if pval_type == "two-tailed":
            # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
            empirical_pval = max(1/len(Wp_XYnull), 2*min(prop_lower, 1 - prop_lower))
        elif pval_type in ["all", "right"]:
            # compute p-value from 1-sided (right) test against empirical CDF (enforcing inf(p)=1/N)
            empirical_pval = max(1/len(Wp_XYnull), 1 - prop_lower)
        elif pval_type == "left":
            # compute p-value from 1-sided (left) test against empirical CDF (enforcing inf(p)=1/N)
            empirical_pval = max(1/len(Wp_XYnull), prop_lower)
        else:
            raise ValueError(f"Unrecognized p-value type {pval_type}")

    if empirical_pval == 1:
        empirical_pval = 1 - 1/len(Wp_XYnull)

    df["empirical_pval"] = [empirical_pval] + [np.nan]*len(Wp_XYnull)
    df["pval_type"] = [pval_type] + [np.nan]*len(Wp_XYnull)

    return df

def correct_pvals(pval_vec, verbose=True, low_thresh=0.01, high_thresh=0.05):
    if verbose:
        print(f"correcting family of {len(pval_vec)} pvals")
        low_count = np.count_nonzero(pval_vec < low_thresh)
        high_count = np.count_nonzero(pval_vec < high_thresh)
        print(f"found {low_count} signficant (<{low_thresh}) pvals before correction")
        print(f"found {high_count} signficant (<{high_thresh}) pvals before correction")

    rmv_idx = np.isnan(pval_vec) + (pval_vec < 0)
    _, corr_pvals = fdrcorrection(pval_vec[ rmv_idx == False ])
    pval_vec[ rmv_idx == False ] = corr_pvals

    if verbose:
        low_count = np.count_nonzero(pval_vec < low_thresh)
        high_count = np.count_nonzero(pval_vec < high_thresh)
        print(f"found {low_count} signficant (<{low_thresh}) pvals after correction")
        print(f"found {high_count} signficant (<{high_thresh}) pvals after correction")

    return pval_vec
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def pull_data(
        fpath_list=None, parent_dir=None, dir_pattern=def_pattern, f_pattern = '*_vs_*', 
        check_pval=True, pval_type="all", data_only=True, debug=False
        ):

    if fpath_list is None and parent_dir is not None:
        parent_pattern = os.path.join(parent_dir, dir_pattern)
        dirlist = glob.glob(parent_pattern)
        dirlist.sort()
        if debug:
            print(f"matching files of with pattern \'{f_pattern}\' in directories matching \'{parent_pattern}\'")

        fpath_grid = [ glob.glob(os.path.join(X_dir, f"{f_pattern}.json")) for X_dir in dirlist ]
    else:
        fpath_grid = [ fpath_list for i in range(len(fpath_list)) ]
        
    [i.sort() for i in fpath_grid]

    if data_only:
        print("Only retaining information from datatype \"Data\" (discarding \"Null\"-type data after necessary computations)")
    
    alldata_grid = [ [ _load(
        fpath, data_only=data_only, check_pval=check_pval, pval_type=pval_type,
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
            gridlist_shape = [len(alldata_grid), set([len(i) for i in alldata_grid]), np.mean([len(i) for j in alldata_grid for i in j])]
        else:
            gridlist_shape = [len(alldata_grid), set([len(i) for i in alldata_grid]), set([i.shape for j in alldata_grid for i in j])]

        print(f"gridlist has \"shape\" given by \n{gridlist_shape}")
        ### debugging code ###

    return alldata_grid


def _load(input_fpath, check_pval=True, data_only=True, parse_longname=False, pval_type="all"):
    with open(input_fpath, 'r') as fin:
        data_df = pd.DataFrame(json.load(fin))

    if parse_longname:
        data_df[["X_mod","X_feat","X_diff"]] = data_df["X_type"].str.split('_', n=2, expand=True)
        data_df[["Y_mod","Y_feat","Y_diff"]] = data_df["Y_type"].str.split('_', n=2, expand=True)
        data_df.drop(["X_type","Y_type"], axis=1, inplace=True)


    if check_pval:
        if "empirical_pval" not in data_df.columns:
            data_df = _add_emp_pval(data_df, pval_type=pval_type)

    if data_only:
        null_df = data_df[data_df["datatype"] == "Null"]
        data_df = data_df[data_df["datatype"] == "Data"]
        data_df["Wp_XYNull_mean"] = np.mean(null_df["Wp_XY"])
        data_df["Wp_XYNull_std"] = np.std(null_df["Wp_XY"])

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
        default=None,
        help="directory with name of type []_vs_[] containing bootstrapped distance outputs"
    )
    parser.add_argument(
        "-F",
        "--fpathlist_path",
        type=str,
        default=None,
        help="filepath to .csv (or .txt) list of filepaths to results to be visualized"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="figure output directory"
    )
    parser.add_argument(
        "-p",
        "--pval_type",
        type=str,
        default="two-tailed",
        help="choose between \'left\', \'right\', or \'two-tailed\' p-value calculation -- or \'all\' to calculate all 3"
    )
    parser.add_argument(
        "-r",
        "--pattern_restriction",
        type=str,
        default=None,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-L",
        "--log_scale",
        default=False,
        action="store_true",
        help="apply log10 to display values (collapse difference)"
    )
    parser.add_argument(
        "-w",
        "--write_mode",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="toggle verbose output"
    )
    args = parser.parse_args()
    
    if args.pattern_restriction is not None:
        args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        dir_pattern=f'*X_*{args.pattern_restriction}*_dists'
        f_pattern = f'*{args.pattern_restriction}*_vs_*{args.pattern_restriction}*'
    else:
        dir_pattern=f'X_*_dists'
        f_pattern = f'*_vs_*'

    if args.verbose:
        var_dict = vars(args)
        for varname in var_dict:
            print(f"The argument \'{varname}\' has been initialized with value: {var_dict[varname]}")
        print(f"The argument \'dir_pattern\' has been initialized with value: {dir_pattern}")
        print(f"The argument \'f_pattern\' has been initialized with value: {f_pattern}")

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)

    debug=False

    if args.fpathlist_path is None: 
        fpath_list = None
    else:
        with open(args.fpathlist_path, 'r') as fin:
            fpath_list = fin.read().split('\n')

    alldata_grid = pull_data(
            fpath_list = fpath_list,
            parent_dir = args.input_dir, 
            dir_pattern= dir_pattern,
            f_pattern =  f_pattern,
            data_only = True,
            check_pval = True,
            pval_type = args.pval_type
            )
    if debug:
        intm_dir = os.path.join(args.output_dir, "alldata_grid")
        if not os.path.isdir(intm_dir):
            os.mkdir(intm_dir)
        for i, sublist in enumerate(alldata_grid):
            for j, df in enumerate(sublist):
                fname = f"alldata_col{i}_row{j}.csv"
                df.to_csv(os.path.join(intm_dir, fname))

    xnamelist, ynamelist, valuegrid = _get_heatmap_inputs(alldata_grid, clustermap_vars=def_clustermap_vars)

    if debug:
        for name in list(valuegrid.keys()):
            savepath = os.path.join(args.output_dir, f"{name}.csv")
            np.savetxt(savepath, valuegrid[name])
            print(f"wrote value grid for value \"{name}\" to \"{savepath}\"")

    fig_inches = def_fig_size[0] * np.sqrt(73 / len(xnamelist))   # calibrating label fontsize to number of entries
    label_fontsize = def_label_fontsize * np.sqrt(73 / len(xnamelist))   # calibrating label fontsize to number of entries

    generate_clustermaps(
            xnamelist, 
            ynamelist, 
            valuegrid, 
            linkage_var = "Wp_XY",
            cluster_method = "average",
            log_scale = args.log_scale,
            fig_size = (fig_inches, fig_inches),
            label_fontsize = label_fontsize,
            outdir=args.output_dir,
            write_mode=args.write_mode
            )
