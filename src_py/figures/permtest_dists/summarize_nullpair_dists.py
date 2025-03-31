import re
import os
import glob
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
def_clustermap_vars = ["Wp_XY", "Wp_XYNull_mean", "Wp_XYNull_std"]
def_scatter_vars = ["Wp_XY", "Y_type"] 

# exp_outtype="All_vs_AllNull/X_ICA15_Amps_Psim_dists/ICA15_Amps_Psim_vs_Schaefer100_Amps_Psim_null-subjectPerms.csv"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

def main(args, debug=True):

    if args.verbose:
        var_dict = vars(args)
        print(f"argument initializations for: \n{__name__}\n***")
        for varname in list(var_dict.keys()):
            print(f"\tThe argument \'{varname}\' has been initialized with value: {var_dict[varname]}")

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)


    if args.fpathlist_path is None: 
        fpath_list, fpath_grid = _get_fpath_sets(args)

    else:
        with open(args.fpathlist_path, 'r') as fin:
            fpath_list, fpath_grid = fin.read().split('\n')

    if args.solo_plots:
        from single_null_dists import make_solo_plots
        args.fig_size=(6,6)
        make_solo_plots(fpath_list, dist_type="pair", figs=args.write_mode, args=args)
        exit()

    alldata_grid = pull_data(
            fpath_grid,
            args,
            data_only = True,
            check_pval = True
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

    return xnamelist, ynamelist, valuegrid



############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty paired-null distance distribution summaries
########################################################################################################################
def one_pair_plot(fpath, fig_title=None, verbose=True, debug=False):
    full_df = pd.read_csv(fpath, index_col=0)
    data_df = full_df[ full_df["datatype"]!="Null" ]
    null_df = full_df[ full_df["datatype"]=="Null" ]

    # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
    data_pval = 1 - np.mean(data_df > null_df["Wp_XY"].to_numpy())
    data_pval = min(data_pval, 1 - data_pval)
    if data_pval < 1/len(null_df):
        data_pval = 1/len(null_df)

    g = sns.displot(data=null_df, x="Wp_XY", kind="hist", kde=True, hue="permtype")
    g.refline(x=data_df, linestyle="--", color="red", label="data distance")

    if fig_title is None:
        X_type = outputs[0]["X_type"]
        Y_type = outputs[0]["Y_type"]
        fig_title = f"{X_type}_vs_{Y_type}\nreal vs. permuted p-Wasserstein distances"
    
    g.fig.suptitle(fig_title)

    if verbose:
        print(f"The approximate empircal p-value for data vs. null distance of {data_df} is {data_pval}")

    return g, data_pval
########################################################################################################################

# heatmap plotting
########################################################################################################################
def _get_heatmap_inputs(alldata_grid, clustermap_vars=def_clustermap_vars, check_pval=True, debug=False):
    xnamelist = [list(set(i[0]["X_type"]))[0] for i in alldata_grid]
    ynamelist = [list(set(j["Y_type"]))[0] for j in alldata_grid[0]]


    valuegrid = {}
    if check_pval:
        pval_vars = [ varname for varname in alldata_grid[0][0].columns.values if "pval" in varname ]
        clustermap_vars = clustermap_vars + pval_vars

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
            if ("fwe-" in linkage_var) and ("fwe-" in display_var):
                print(f"Skipping \"cluster {display_var} on {linkage_var}\" plot.")
                continue
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
        display_var, display_vals, ttl_suffix = _disp_logdata(display_var, display_vals)
        cm_title = cm_title + ttl_suffix


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
def _disp_logdata(varname, values, disp_var=True):
    if disp_var and "pval" in varname:
        if "fdr" in varname:
            values = squareform(correct_pvals(triu_vals(values,k=1), corr_type="fdr"))
        elif "fwe" in varname:
            values = squareform(correct_pvals(triu_vals(values,k=1), corr_type="fwe"))
        np.fill_diagonal(values, np.nan)
        title_suffix = " (-log10(2p))"
        values = -np.log10(2*values)
        nanval = -1

    if "Wp_XY" in varname:
        values[ values==0 ] = np.nan
        title_suffix = " (log10(W_p))"
        values= np.log10(values)
        nanval = -1.1*np.nanmax(np.abs(values))

    print(f"replacing NaNs in{title_suffix} for {varname} with {nanval}")
    np.nan_to_num(values, nan=nanval, copy=False)
    varname = f"log-{varname}"
    return varname, values, title_suffix
########################################################################################################################
########################################################################################################################


# compute secondary statistics
########################################################################################################################
# add an empirical p-val to a dataframe containing only a single data-derived distance and its null counterparts
def _add_emp_pval(df, check_match=True, tail_type="all", corr_type="fdr", null_hi=None, null_lo=None, debug=False):
    datarow = df[df["datatype"] == "Data"]

    Wp_XY = datarow["Wp_XY"].to_numpy()
    if corr_type == "fdr":
        nullrows = df[df["datatype"] == "Null"]
        null_lo = nullrows["Wp_XY"].to_numpy()
        null_hi = nullrows["Wp_XY"].to_numpy()
        check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
        err_str =  f"data row and null rows are not of matching type: \n{[[col, set(datarow[col]), set(nullrows[col])] for col in check_cols]}"
        assert all( [ set(datarow[col]) == set(nullrows[col]) for col in check_cols ] ), err_str

    err_msg = "extremal family-wise distributions must be provided for family-wise error (\'fwe\') p-value correction"
    assert null_hi is not None and null_lo is not None, err_msg

    if debug:
        print(f"Adding p-value of type {tail_type} with multiple comparison correction method {corr_type}")

    if tail_type == "all":
        tails = ["left", "right", "two-tailed"]
    else:
        tails = [tail_type]

    try:
        epsilon_lo = 1/len(null_lo)
        epsilon_hi = 1/len(null_hi)
        p_lo = 1 - np.mean(Wp_XY <= null_lo)
        p_hi = 1 - np.mean(Wp_XY >= null_hi)

        for tail in tails:
            if tail == "two-tailed":
                # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = 2*min( max(epsilon_lo, p_lo), max(epsilon_hi, p_hi) )
            elif tail == "right":
                # compute p-value from 1-sided (right) test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = min( max(epsilon_hi, p_hi), 1 - epsilon_hi )
            elif tail == "left":
                # compute p-value from 1-sided (left) test against empirical CDF (enforcing inf(p)=1/N)
                empirical_pval = min( max(epsilon_lo, p_lo), 1 - epsilon_lo )
            else:
                raise ValueError(f"Unrecognized p-value type {tail}")

            df[f"{corr_type}-{tail}-pval"] = empirical_pval
    except Exception as err:
        for tail in tails:
            print(f"p-value computation failed for correction type {corr_type} with tail type {tail}: \n{err}")
            df[f"{corr_type}-{tail}-pval"] = np.nan

    return df

def correct_pvals(pval_vec, verbose=True, low_thresh=0.01, high_thresh=0.05, corr_type="fdr"):
    low_count = np.count_nonzero(pval_vec < low_thresh)
    high_count = np.count_nonzero(pval_vec < high_thresh)
    if corr_type == "fdr":
        if verbose:
            print(f"correcting family of {len(pval_vec)} pvals")
            print(f"found {low_count} signficant (<{low_thresh}) pvals before correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals before correction")

        rmv_idx = np.isnan(pval_vec) + (pval_vec < 0)
        _, corr_pvals = fdrcorrection(pval_vec[ rmv_idx == False ])
        pval_vec[ rmv_idx == False ] = corr_pvals

        if verbose:
            low_count = np.count_nonzero(pval_vec < low_thresh)
            high_count = np.count_nonzero(pval_vec < high_thresh)
            print(f"found {low_count} signficant (<{low_thresh}) pvals after \'fdr\' correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals after \'fdr\' correction")
    elif corr_type == "fwe":
        if verbose:
            print(f"found {low_count} signficant (<{low_thresh}) pvals after \'fwe\' correction")
            print(f"found {high_count} signficant (<{high_thresh}) pvals after \'fwe\' correction")


    return pval_vec
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def pull_data(
        fpath_grid, args, check_pval=True, data_only=True, debug=True
        ):

    if data_only:
        print("Only retaining information from datatype \"Data\" (discarding \"Null\"-type data after necessary computations)")

    if args.corr_type == "fwe":
        null_lo, null_hi = _pull_extremal_dists(args)
    else:
        null_lo = None
        null_hi = None
    
    alldata_grid = [ 
                    [ _load(
        fpath, data_only=data_only, permtype=args.permtype,
        check_pval=check_pval, tail_type=args.tail_type, 
        corr_type=args.corr_type, null_hi=null_hi, null_lo=null_lo,
        ) for fpath in X_sublist ] 
                    for X_sublist in fpath_grid ]

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


def _load(
        input_fpath, data_only=True, parse_longname=False, permtype="feature",
        check_pval=True, tail_type="all", corr_type="fwe", null_hi=None, null_lo=None
        ):
    data_df = pd.read_csv(input_fpath, index_col=0)

    if parse_longname:
        data_df[["X_mod","X_feat","X_diff"]] = data_df["X_type"].str.split('_', n=2, expand=True)
        data_df[["Y_mod","Y_feat","Y_diff"]] = data_df["Y_type"].str.split('_', n=2, expand=True)
        data_df.drop(["X_type","Y_type"], axis=1, inplace=True)


    if check_pval:
        if not any(["pval" in col for col in data_df.columns]):
            data_df = _add_emp_pval(data_df, tail_type=tail_type, corr_type=corr_type, null_hi=null_hi, null_lo=null_lo)

    if data_only:
        null_df = data_df[data_df["datatype"] == "Null"]
        data_df = data_df[data_df["datatype"] != "Null"]
        try:
            data_df["Wp_XYNull_mean"] = np.mean(null_df["Wp_XY"])
            data_df["Wp_XYNull_std"] = np.std(null_df["Wp_XY"])
            data_df["permtype"] = '-'.join(list(set(null_df["permtype"])))
            data_df.drop(["permlabel"], axis=1, inplace=True)
        except KeyError as err:
            print(f"Encountered KeyError! offending dataframe has \n\'data_df\'=\n{data_df}\n and \n\'null_df\'=\n{null_df}\n")
            print(f"loaded from filepath: \n{input_fpath}")
            exit()

    return data_df

def _get_fpath_sets(args, debug=False):
    if args.pattern_restriction is not None and args.permtype is not None:
        if not args.output_dir.endswith(args.pattern_restriction):
            args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)

        args.dir_pattern=f'*X_*{args.pattern_restriction}*_dists'
        args.f_pattern = f'*{args.pattern_restriction}*_vs_*{args.pattern_restriction}*{args.permtype}Perms.csv'
    elif args.permtype is not None:
        args.dir_pattern=f'X_*'
        args.f_pattern = f'*_vs_*{args.permtype}Perms.csv'
    else:
        args.dir_pattern=f'X_*'
        args.f_pattern = f'*_vs_*.csv'

    pdir_pattern = os.path.join( args.input_dir, args.dir_pattern )

    fpath_grid = [ glob.glob(os.path.join(dpath, args.f_pattern)) for dpath in glob.glob(pdir_pattern) ]
    fpath_grid = [ pathlist for pathlist in fpath_grid if pathlist ]    # removes empty lists (corresponding to directories with no successful search hits)
    [pathlist.sort() for pathlist in fpath_grid]
    
    fpath_list = list(itertools.chain(*fpath_grid))

    if args.verbose:
        print(f"matching patterns of general form: \n{(pdir_pattern, args.f_pattern)}")
        print(f"shaping matches into a \'filepath grid\' array results in shape(s): \n{ ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
        print(f"found {len(fpath_list)} total matches.")

    if debug:
        import json
        with open("fpath_grid_tmp.txt", 'w') as fout:
            json.dump(fpath_grid, fout, indent=4)

    return fpath_list, fpath_grid

def _pull_extremal_dists(args):
    import extremal_nullpair_dists as ex_null

    args.extrema_only = True
    args.verbose = False
    extrema_df,_ = ex_null.main(args)
    return extrema_df["Wp_XYNull_min"].values, extrema_df["Wp_XYNull_max"].values


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
        "--tail_type",
        type=str,
        default="all",
        help="choose between \'left\', \'right\', or \'two-tailed\' p-value calculation -- or \'all\' to calculate all 3. supports both \'fwe\' and \'fdr\' pval correction types."
    )
    parser.add_argument(
        "-c",
        "--corr_type",
        type=str,
        default="fwe",
        help="choose family-wise error (\'fwe\') or false discovery rate (\'fdr\') pval correction types."
    )
    parser.add_argument(
        "-P",
        "--permtype",
        type=str,
        default="feature",
        help="permutation type: either \'subject\' or \'feature\'"
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
        "-S",
        "--solo_plots",
        default=False,
        action="store_true",
        help="flag to perform solo plots"
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
    
    xnamelist, ynamelist, valuegrid = main(args)

    fig_inches = def_fig_size[0] * np.sqrt(73 / len(xnamelist))   # calibrating label fontsize to number of entries
    label_fontsize = def_label_fontsize * np.power(73 / len(xnamelist), 3/4)   # calibrating label fontsize to number of entries

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
