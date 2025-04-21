import re
import os
import glob
import copy
import scipy
import argparse
import itertools
import functools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import figstats as fstats
import figutils as futils
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import fdrcorrection

# global variables 

def_fig_size = (24, 24)
def_label_fontsize = 7 
def_pattern='*X_*_dists'

# cmap_list = ["#808080", ("#ffffff", 0.0)]         # color values to match to False/True
# cmap_in = LinearSegmentedColormap.from_list( 'mask_overlay', cmap_list )

# def_clustermap_vars = ["Wp_XY", "empirical_pval"]
# def_scatter_vars = ["Wp_XY", "Y_name"] 

def_clustermap_vars = ["Wp_XY"]
# def_clustermap_vars = ["Wp_XY", "Wp_XYNull_mean", "Wp_XYNull_std"]

# exp_outtype="All_vs_AllNull/X_ICA15_Amps_Psim_dists/ICA15_Amps_Psim_vs_Schaefer100_Amps_Psim_null-subjectPerms.csv"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

def main(args, debug=False):

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
            fpath_list = fin.read().split('\n')

    if args.clustermap_plots:
        xnamelist, ynamelist, value_set = make_clustermaps(fpath_grid, args=args)

    if args.solo_plots:
        from single_null_dists import make_solo_plots
        args.fig_size=(6,6)
        make_solo_plots(fpath_list, dist_type="pair", args=args)
        exit()

    if args.distribution_plots:
        from single_null_dists import make_distribution_plots
        args.fig_size=(12,12)
        make_distribution_plots(fpath_list, dist_type="pair", args=args) 

    return xnamelist, ynamelist, value_set



############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty paired-null distance distribution summaries (defunct)
########################################################################################################################
def one_pair_plot(fpath, fig_title=None, verbose=True, debug=False):
    full_df = pd.read_csv(fpath, index_col=0)
    null_mask = full_df["datatype"].str.contains("Null")
    data_df = full_df[ ~null_mask ]
    null_df = full_df[ null_mask ]

    # compute p-value from two-tailed test against empirical CDF (enforcing inf(p)=1/N)
    data_pval = 1 - np.mean(data_df > null_df["Wp_XY"].to_numpy())
    data_pval = min(data_pval, 1 - data_pval)
    if data_pval < 1/len(null_df):
        data_pval = 1/len(null_df)

    g = sns.displot(data=null_df, x="Wp_XY", kind="hist", kde=True, hue="permtype")
    g.refline(x=data_df, linestyle="--", color="red", label="data distance")

    if fig_title is None:
        X_name = outputs[0]["X_name"]
        Y_name = outputs[0]["Y_name"]
        fig_title = f"{X_name}_vs_{Y_name}\nreal vs. permuted p-Wasserstein distances"
    
    g.fig.suptitle(fig_title)

    if verbose:
        print(f"The approximate empircal p-value for data vs. null distance of {data_df} is {data_pval}")

    return g, data_pval


def make_clustermaps(fpath_grid, args=None, debug=False):
    alldata_grid = pull_data(
            fpath_grid,
            args,
            check_pval = True
            )
    if args is None:
        alpha = None
    else:
        alpha = args.alpha

    if debug:
        intm_dir = os.path.join(args.output_dir, "alldata_grid")
        if not os.path.isdir(intm_dir):
            os.mkdir(intm_dir)
        for i, sublist in enumerate(alldata_grid):
            for j, df in enumerate(sublist):
                fname = f"alldata_col{i}_row{j}.csv"
                df.to_csv(os.path.join(intm_dir, fname))

    xnamelist, ynamelist, value_set = _get_heatmap_inputs(alldata_grid, clustermap_vars=def_clustermap_vars)

    if debug:
        for name in list(value_set.keys()):
            savepath = os.path.join(args.output_dir, f"{name}.csv")
            np.savetxt(savepath, value_set[name])
            print(f"wrote value grid for value \"{name}\" to \"{savepath}\"")
    
    fig_inches = def_fig_size[0] * np.sqrt(70 / len(xnamelist))   # calibrating label fontsize to number of entries
    label_fontsize = def_label_fontsize * np.power(70 / len(xnamelist), 3/4)   # calibrating label fontsize to number of entries

    generate_clustermaps(
            xnamelist, 
            ynamelist, 
            value_set, 
            linkage_var = "Wp_XY",
            cluster_method = "average",
            alpha = args.alpha,
            log_scale = args.log_scale,
            fig_size = (fig_inches, fig_inches),
            label_fontsize = label_fontsize,
            outdir=args.output_dir,
            write_mode=args.write_mode
            )

    return xnamelist, ynamelist, value_set

########################################################################################################################

# heatmap plotting
########################################################################################################################
def _get_heatmap_inputs(alldata_grid, clustermap_vars=def_clustermap_vars, check_pval=True, debug=True):
    xnamelist = [i[0]["X_name"].unique()[0] for i in alldata_grid]
    ynamelist = [j["Y_name"].unique()[0] for j in alldata_grid[0]]

    value_set = {}
    if check_pval:
        pval_vars = [ varname for varname in alldata_grid[0][0].columns.values if "pval" in varname ]
        clustermap_vars = clustermap_vars + pval_vars

    for varname in clustermap_vars:
        try:
            vals = np.squeeze(np.array([[j[varname].to_numpy() for j in i] for i in alldata_grid]))
            if debug:
                print(f"variable {varname} has grid of values with shape: {vals.shape}")
        except ValueError:
            new_entry = [[j[varname].to_numpy() for j in i] for i in alldata_grid]
            if debug:
                ### debugging code ###
                shape_vec = [ [ (len(var), i.shape) for i in var ] for var in new_entry ]
                name_grid = np.array([[(j["X_name"].unique().astype(str), j["Y_name"].unique().astype(str)) for j in i] for i in alldata_grid])
                futils._write_list("debug/shape_vec.txt", shape_vec)
                futils._write_list("debug/name_grid.txt", name_grid)
                print(f"found data inhomogeneity in {varname} readin. Saved grid of shapes and pair names to (resp.) paths:")
                print("debug/shape_vec.txt")
                print("debug/name_grid.txt")
                exit()
                ### debugging code ###

        value_set[varname] = vals
        print(f"\'{varname}\' gridded.")

    if debug:
        ### debugging code ###
        print(f"Names of {len(xnamelist)} 'X' spaces: \n{xnamelist}")
        print(f"Names of {len(ynamelist)} 'Y' spaces: \n{ynamelist}")
        print(f"Entries in list of grid values have the following shapes: \n{[value_set[var].shape for var in list(value_set.keys())]}")
        # print("First entry in value_set: ", np.array(value_set[clustermap_vars[0]]))
        print(f"Generating one heatmap for each of the following set of variables: \n{list(value_set.keys())}")
        print("")
        ### debugging code ###
    return xnamelist, ynamelist, value_set


def generate_clustermaps(
        xnamelist,
        ynamelist,
        value_set,
        onelink = True,
        linkage_var = "Wp_XY",
        cluster_method = "average",
        alpha = None,
        log_scale = True,
        fig_size = def_fig_size,
        label_fontsize = def_label_fontsize,
        outdir = None,
        write_mode = True
        ):
    dispvars = list(value_set.keys())

    if onelink:
        assert linkage_var in dispvars, f"Value does not include variable \"{linkage_var}\", the specified common linkage operator"
        print(f"Using \"{linkage_var}\" as linkage variable while generating clustermaps")
        linkvars = [linkage_var]
    else:
        print(f"Plotting clustermaps for all (linkage_var, display_var) value pairs (including self-pairs) in {dispvars}")
        linkvars = dispvars

    fig_dict = {}

    if alpha is not None:
        # retains only display grid values corresponding to significant p-values
        pval_vars = [var for var in dispvars if "pval" in var]
        left_pval = [var for var in pval_vars if "left" in var][0]

        # create logical significance arrays
        sig_list = [ _enforce_symmetry(value_set[var], fill_val = 1) < alpha for var in pval_vars]
        
        # define significance array for sig_list U sig_right
        pval_vars.append(left_pval.replace("left","left-right"))
        sig_list.append( sig_list[pval_vars.index(left_pval)] | sig_list[pval_vars.index(left_pval.replace("left","right"))] )

        # 'mask' (in sns.heatmap) hides values at coordinate if mask(coord)=True; retain values by setting mask(coord)=False.
        masks = [ ~sig for sig in sig_list ]
    else:
        pval_vars = [None]
        masks = [None]

    dispvars = [var for var in dispvars if var not in pval_vars]

    for linkage_var in linkvars:
        for display_var in dispvars:
            for pval_var in pval_vars:
                mask = masks[pval_vars.index(pval_var)]
                if ("pval" in linkage_var) and ("pval" in display_var):
                    print(f"Skipping \"cluster {display_var} on {linkage_var}\" plot.")
                    continue
                fig_dict[display_var] = plot_clustermap(
                        xnamelist,
                        ynamelist,
                        value_set,
                        cluster_method = cluster_method,
                        linkage_var = linkage_var,
                        display_var = display_var,
                        alpha = alpha,
                        pval_var = pval_var,
                        mask = mask,
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
        value_set,
        cluster_method = "average",
        linkage_var = "Wp_XY",
        display_var = "empirical_pval",
        alpha = None,
        pval_var = None,
        mask = None,
        enf_sym = False,
        log_scale = True,
        label_fontsize = def_label_fontsize,
        fig_size = def_fig_size,
        outdir = None,
        write_mode = True,
        debug = False
        ):

    print(f"enforcing symmetry in \'{linkage_var}\' linkage values.")
    linkage_vals = _enforce_symmetry(value_set[linkage_var], fill_val=0)
    import scipy.cluster.hierarchy as hc
    xlinkage = hc.linkage(squareform(linkage_vals), method=cluster_method, optimal_ordering=True)

    if debug:
        print(f"found {np.count_nonzero(xlinkage < 0)} negative linkage values") 
        print(f"found {np.count_nonzero(np.isnan(xlinkage))} NaN linkage values")
        print(f"found {np.count_nonzero(np.isinf(xlinkage))} infinite linkage values")

    if enf_sym:
        print(f"enforcing symmetry in \'{display_var}\' display values.")
        display_vals = _enforce_symmetry(value_set[display_var], fill_val=0)
        try:
            assert xnamelist == ynamelist
        except AssertionError:
            if debug:
                print(f"namelists are unequal in forced symmetric case! xnamelist: {len(xnamelist)} entries, ynamelist: {len(ynamelist)} entries")
                # print(f"namelists are unequal in forced symmetric case! \nxnamelist: {len(xnamelist)} entries\nynamelist: {len(ynamelist)} entries")
            ynamelist = xnamelist
    else:
        display_vals = value_set[display_var].copy()

    assert linkage_vals.shape==display_vals.shape, "linkage and display values must have same dimensions!"
    
    print(f"Plotting grid of '{display_var}' values...")

    xticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in xnamelist]
    yticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in ynamelist]
    
    cm_title = f"Clustermap plot of {display_var} \n(clustered on {linkage_var})"


    if log_scale:
        display_var, display_vals, ttl_suffix = _disp_logdata(display_var, display_vals)
        cm_title = cm_title + ttl_suffix


    if (mask is not None) and ("pval" not in display_var):
        print(f"Masking \'{display_var}\' plot by \'{pval_var}\'...")
        outname = f"cluster-on-{linkage_var}_of-{display_var}_mask-{pval_var}_alpha{alpha}.png".replace(" ","").replace("0.","")
    else:
        mask = None
        outname = f"cluster-on-{linkage_var}_of-{display_var}.png".replace(" ","")

        
    if np.count_nonzero(np.isnan(display_vals)) > 0:
        if debug:
            print(f"{np.count_nonzero(np.isnan(display_vals))} NaNs removed removed from \'display_vals\' for var \"{display_var}\"")
        np.nan_to_num(display_vals, nan=-1, copy=False)

#   if debug:
#       ### debugging code ###
#       print(f"xticklabels: {xticklabels[0]}")
#       print(f"yticklabels: {yticklabels[0]}")

    from compare_topostats import _plot_clustermap as _pcl

    g = _pcl(
        display_vals, 
        cluster=True,
        cluster_method=cluster_method,
        cm_title = cm_title,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        xlinkage=xlinkage,
        ylinkage=xlinkage,
        mask = mask,
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
        outpath = os.path.join(outdir, outname)
        futils._write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()

    return g

# heatmap plot utility
def _disp_logdata(varname, values, disp_var=True):
    if disp_var and "pval" in varname:
        if "fdr" in varname:
            values = squareform(fstats.correct_pvals(triu_vals(values,k=1), corr_type="fdr"))
        elif "fwe" in varname:
            values = squareform(fstats.correct_pvals(triu_vals(values,k=1), corr_type="fwe"))
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



# Data wrangling functions
########################################################################################################################
def pull_data(
        fpath_grid, args, check_pval=True, debug=False
        ):

    if args.corr_type == "fwe":
        null_lo, null_hi = _pull_extremal_dists(args)
    else:
        null_lo = None
        null_hi = None

    pv_args = copy.deepcopy(args)
    pv_args.null_lo = null_lo
    pv_args.null_hi = null_hi
    
    alldata_grid = [ [ futils._load(
        fpath, 
        load_type="pair",
        permtype=args.permtype,
        check_pval=check_pval,
        pval_args = pv_args
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


def _get_fpath_sets(args, debug=False):
    if args.pattern_restriction is not None and args.permtype is not None:
        if not args.output_dir.endswith(args.pattern_restriction):
            args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)

        args.dir_pattern=f'*X_*{args.pattern_restriction}*_dists'
        args.f_pattern = f'*{args.pattern_restriction}*_vs_*{args.pattern_restriction}*{args.permtype}Perms.csv'
    elif args.permtype is not None:
        args.dir_pattern='X_*'
        args.f_pattern = f'*_vs_*{args.permtype}Perms.csv'
    else:
        args.dir_pattern='X_*'
        args.f_pattern = '*_vs_*.csv'

    pdir_pattern = os.path.join( args.input_dir, args.dir_pattern )

    dpath_list = glob.glob(pdir_pattern); dpath_list.sort()
    fpath_grid = [ glob.glob(os.path.join(dpath, args.f_pattern)) for dpath in dpath_list ]
    fpath_grid = [ pathlist for pathlist in fpath_grid if pathlist ]    # removes empty lists (corresponding to directories with no successful search hits)
    [pathlist.sort() for pathlist in fpath_grid]
    
    fpath_list = list(itertools.chain(*fpath_grid))

    if args.verbose:
        print(f"Matching patterns of general form: \n{(pdir_pattern, args.f_pattern)}:")
        print(f"\tshaping matches into a \'filepath grid\' array results in shape(s): { ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
        print(f"\tfound {len(fpath_list)} total matches.")

    if debug:
        import json
        with open("fpath_grid_tmp.txt", 'w') as fout:
            json.dump(fpath_grid, fout, indent=4)

    xnamelist = [_semiload(i[0])["X_name"].unique()[0] for i in fpath_grid]
    ynamelist = [_semiload(j)["Y_name"].unique()[0] for j in fpath_grid[0]]

    if (args.alpha is not None) and any( [args.clustermap_plots, args.solo_plots, args.distribution_plots] ):
        fpath_grid = _filter_fpath_grid(args, fpath_grid)
        fpath_list = list(itertools.chain(*fpath_grid))
        if args.verbose:
            print(f"After apply AUC filtering at significance threshold alpha={args.alpha}:")
            print(f"\tshaping matches into a \'filepath grid\' array results in shape(s): { ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
            print(f"\tfound {len(fpath_list)} total matches.")


    return fpath_list, fpath_grid

def _filter_fpath_grid(args, fpath_grid, backend="text"):
    if args.verbose:
        print("Filtering by AUC significance:")
    auc_mask = futils._get_auc_mask(args=args)

    if backend=="text":
        call_mask = lambda x: auc_mask[x]
        fpath_grid = [ [ fpath for fpath in fpathlist 
                        if all(map(call_mask, futils._parse_fpath(fpath, pathtype="pair")))
                        ] for fpathlist in fpath_grid ]                         # removes filepath if either the X_name or Y_name fail significance
        fpath_grid = [ fpathlist for fpathlist in fpath_grid if fpathlist ]     # removes empty filepath sublists
    elif backend=="grid":
        _, _, fpath_grid = futils._apply_series_mask(auc_mask, xnamelist, ynamelist, fpath_grid)

    return fpath_grid

def _pull_extremal_dists(args):
    import extremal_nullpair_dists as ex_null

    args.extrema_only = True
    args.verbose = False
    extrema_df,_ = ex_null.main(args)
    return extrema_df["Wp_XYNull_min"].values, extrema_df["Wp_XYNull_max"].values

def _semiload(fpath):
    df = pd.read_csv(fpath, index_col=0)
    df = futils._unify_df(df)
    return df

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
        default="subject",
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
        "-D",
        "--distribution_plots",
        default=False,
        action="store_true",
        help="flag to visualized grouped distributions"
    )
    parser.add_argument(
        "-C",
        "--clustermap_plots",
        default=False,
        action="store_true",
        help="flag to visualized grouped distributions"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        default=None,
        type=float,
        help="significance threshold"
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
    
    xnamelist, ynamelist, value_set = main(args)

