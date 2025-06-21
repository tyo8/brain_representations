import re
import os
import glob
import copy
import scipy
import argparse
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import figstats as fstats
import figutils as futils
from numbers import Number
# from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb

# submodules for different plot types
import barplots
import stackplots
import clustermaps
import scatterplots

# cmap_list = ["#808080", ("#ffffff", 0.0)]         # color values to match to False/True
# cmap_in = LinearSegmentedColormap.from_list( 'mask_overlay', cmap_list )

# def_clustermap_vars = ["Wp_XY", "empirical_pval"]
# def_scatter_vars = ["Wp_XY", "Y_name"] 

# def_clustermap_vars = ["Wp_XY", "Wp_XYNull_mean", "Wp_XYNull_std"]

# exp_outtype="All_vs_AllNull/X_ICA15_Amps_Psim_dists/ICA15_Amps_Psim_vs_Schaefer100_Amps_Psim_null-subjectPerms.csv"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

def main(args, debug=False):

    if args.verbose:
        var_dict = vars(args)
        print(f"argument initializations for: \n{__name__}\n***")
        for varname in list(var_dict.keys()):
            print(f"\tThe argument \'{varname}\' has been initialized with value: {var_dict[varname]}")


    fpath_list, fpath_grid = _get_fpath_sets(args)

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)

    alldata_grid = pull_data(
            fpath_grid,
            args,
            check_pval = True
            )

    if args.clustermap_plots:
        args.fig_size=(24,24)
        xnamelist, ynamelist, value_set, alldata_grid = clustermaps.make(alldata_grid=alldata_grid, args=args)

        if args.scatter_plots:
            args.fig_size=(12,12)
            _, _, value_set = scatterplots.make(alldata_grid=alldata_grid, args=args)
    elif args.scatter_plots:
        args.fig_size=(12,12)
        xnamelist, ynamelist, value_set = scatterplots.make(alldata_grid=alldata_grid, args=args)

    if args.chi2_statistics:
        args.fig_size=(12,6)
        stackplot_df = stackplots.make(value_set, args=args)
        fstats.stackplot_chisq(stackplot_df, args=args)
        args.fig_size=(12,12)
        chisq_results = fstats.make_chisq_summaries(value_set, args=args)
        barplots.make( chisq_results, args=args )

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
########################################################################################################################

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
    fpath_grid = futils._get_fpath_set(args, dist_type="pair", set_type="grid")
    fpath_list = list(itertools.chain(*fpath_grid))

    if args.verbose:
        print(f"Matching patterns of general form: \n{(args.pdir_pattern, args.f_pattern)}:")
        print(f"\tshaping matches into a \'filepath grid\' array results in shape(s): { ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
        print(f"\tfound {len(fpath_list)} total matches.")

    if not len(fpath_list):
        print(f"No filepaths matched search criteria. Exiting.")
        exit()

    if debug:
        import json
        with open("fpath_grid_tmp.txt", 'w') as fout:
            json.dump(fpath_grid, fout, indent=4)

    # xnamelist = [_semiload(i[0])["X_name"].unique()[0] for i in fpath_grid]
    # ynamelist = [_semiload(j)["Y_name"].unique()[0] for j in fpath_grid[0]]

    if (args.alpha is not None) and any( [args.clustermap_plots, args.scatter_plots, args.solo_plots, args.distribution_plots] ):
        fpath_grid = _filter_fpath_grid(args, fpath_grid)
        fpath_list = list(itertools.chain(*fpath_grid))
        if args.verbose:
            print(f"After apply AUC filtering at significance threshold alpha={args.alpha}:")
            print(f"\tshaping matches into a \'filepath grid\' array results in shape(s): { ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
            print(f"\tfound {len(fpath_list)} total matches.")
        if not len(fpath_list):
            print(f"No data pass AUC significance criteria under the given specifications. Exiting.")
            exit()

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
        "-C",
        "--clustermap_plots",
        default=False,
        action="store_true",
        help="flag to visualized grouped distributions"
    )
    parser.add_argument(
        "-V",
        "--scatter_plots",
        default=False,
        action="store_true",
        help="flag to perform solo plots"
    )
    parser.add_argument(
        "-X",
        "--chi2_statistics",
        default=False,
        action="store_true",
        help="flag to perform and visualize post-hoc chi2 testing"
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
        "-a",
        "--alpha",
        default=None,
        type=float,
        help="significance threshold"
    )
    parser.add_argument(
        "-j",
        "--jitter",
        default=0.02,
        type=float,
        help="plot jitter (scatter points uniformly at random to avoid overplotting) -- maximum absolute value of (1-jitter_multiplier)"
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

