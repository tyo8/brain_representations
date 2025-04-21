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
import figstats as fstats
import figutils as futils
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

dist_vars = ["Wp_XYNull_min", "Wp_XYNull_max", "Wp_XYNull_mean", "Wp_XYNull_std", "PDX_diag", "PDY_diag"]
# dist_vars = ["Wp_XYNull_min", "Wp_XYNull_max", "Wp_XYNull_mean", "Wp_XYNull_std", "permtype"]

def main(args, debug=False):
    if args.pattern_restriction is not None:
        if not args.output_dir.endswith(args.pattern_restriction):
            args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
        args.dir_pattern=f'*X_*{args.pattern_restriction}*_dists'
        args.f_pattern = f'*{args.pattern_restriction}*_vs_*{args.pattern_restriction}*.csv'
    else:
        args.dir_pattern=f'X_*_dists'
        args.f_pattern = f'*_vs_*.csv'

    if args.permtype is not None:
        args.f_pattern = args.f_pattern.replace(".csv",f"{args.permtype}Perms.csv")

    if args.fpathlist_path is None: 
        args.search_pattern = os.path.join(args.input_dir, args.dir_pattern, args.f_pattern)
        fpath_list = glob.glob(args.search_pattern)
    else:
        with open(args.fpathlist_path, 'r') as fin:
            fpath_list = fin.read().split('\n')

    if args.verbose:
        print(f"argument initializations for: \n{__name__}\n***")
        var_dict = vars(args)
        for varname in list(var_dict.keys()):
            print(f"The argument \'{varname}\' has initial value: {var_dict[varname]}")
        print(f"Loading extremal null data from {len(fpath_list)} filepaths.")

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)

    df_list = [ futils._load(
        fpath, 
        load_type="ext",
        permtype=args.permtype, 
        debug=False,
        extrema_only=args.extrema_only) for fpath in fpath_list ]   # NOTE: MANUAL INDEXING IS ONLY FOR DEBUGGING PURPOSES
    df = pd.concat(df_list, ignore_index=True)

    if debug:
        check_out=os.path.join(os.getcwd(), "tmp_df.csv")
        print(f"saving dataframe to: \n{check_out}")
        df.to_csv(check_out)

    if args.verbose:
        print(f"collected summary dataframe from {len(df_list)} (sub-)dataframes: \n{df}")
#         for colname in df.columns.values:
#             print(f"Number of unique values in column \'{colname}\' is {len(df[colname].unique())}")

    if args.extrema_only:
        args.dist_vars = ["Wp_XYNull_min", "Wp_XYNull_max"]

    return df, args

def plot_dists(df, args):
    one_displot(
            df,
            dist_vars=args.dist_vars,
            log_scale=args.log_scale,
            extrema_only=args.extrema_only,
            write_mode=args.write_mode, 
            outdir=args.output_dir
            )
    if args.extrema_only:
        for ptype in df["permtype"].unique():
            df_perm = df[ df["permtype"]==ptype ]
            one_displot(
                    df_perm,
                    dist_vars=args.dist_vars,
                    log_scale=args.log_scale,
                    extrema_only=args.extrema_only,
                    write_mode=args.write_mode, 
                    outdir=args.output_dir
                    )
    else:
        for dist_var in args.dist_vars:
            for var in ["permtype", "modality", "feature", "metric"]:
                if not var=="permtype":
                    df[var] = df.apply(lambda x: frozenset([x[f"X_{var}"], x[f"Y_{var}"]]), axis=1)
                one_displot(
                        df,
                        x_var=dist_var,
                        hue_var=var,
                        log_scale=args.log_scale,
                        extrema_only=args.extrema_only,
                        write_mode=args.write_mode, 
                        outdir=args.output_dir
                        )
                if var in ["metric", "feature"]:
                    row_var = f"X_{var}"
                    col_var = f"Y_{var}"
                    one_displot(
                            df,
                            x_var=dist_var,
                            row_var=row_var,
                            col_var=col_var,
                            log_scale=args.log_scale,
                            extrema_only=args.extrema_only,
                            write_mode=args.write_mode, 
                            outdir=args.output_dir
                            )

# make quick and dirty nulled-null distance distribution summaries
########################################################################################################################
def one_displot(
        df,
        dist_vars=dist_vars,
        x_var=None,
        y_var=None,
        row_var=None,
        col_var=None,
        hue_var=None,
        fig_title=None,
        fig_size=None,
        extrema_only=False,
        legend=True,
        log_scale=True,
        regularize=True,
        epsilon=1e-9,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=True
        ):

    if x_var not in df.columns.values and x_var is not None:
        print(f"distribution variable {x_var} not infiltered dataframe. Skipping.")
        return None

    if debug:
        print(f"unique values of permtype: {df.permtype.unique()}")

    if log_scale and regularize:
        if x_var is not None:
            if isinstance(df[x_var].values, np.ndarray):
                df[x_var] = df[x_var] + epsilon
        if y_var is not None:
            if isinstance(df[y_var].values, np.ndarray):
                log_scale = [10, 10]
                df[y_var] = df[y_var] + epsilon
            else:
                y_num = len(df[y_var].unique())
                legend = y_num <= 100
                log_scale = [10, None]

    if hue_var is not None:
        hue_num = len(df[hue_var].unique())
        legend = hue_num <= 20
        if debug:
            ### debugging code ###
            unq_vals = df[hue_var].unique()
            if any(['e-12' in name for name in unq_vals]):
                print(f"unique set of values in dataframe column \'{hue_var}\': \n{unq_vals}")
                err_out = os.path.join(os.getcwd(), "ERR_df.csv")
                print(f"Found epsilon value embedded as qualitive variable label! Saving offending dataframe to: \n{err_out}")
                df.to_csv(err_out)
                print("Exiting.")
                exit()
            ### debugging code ###


    if y_var is None:
        if x_var is None:
            plot_df = df.filter(dist_vars, axis=1).copy()
            plot_df.dropna( how="all" )
            if log_scale and regularize:
                for name in dist_vars:
                    if name in df.columns.values:
                        plot_df[name] = plot_df[name] + epsilon 
            if extrema_only:
                element = 'bars'; multiple = 'dodge'
            else:
                element = 'poly'; multiple = 'layer'
            if verbose:
                print(f"wide-form plotting histogram of dataframe df=\n{plot_df}")
            g = sns.displot(data=plot_df, multiple=multiple, log_scale=log_scale, rug=False, legend=legend, element=element)
        else:
            # g = sns.displot(df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
            g = sns.displot(df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="stack", log_scale=log_scale, rug=False, legend=legend, element='step')
            # g = sns.displot(df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="layer", log_scale=log_scale, rug=False, legend=legend, element='poly')
    else:
        # g = sns.displot(df, x=x_var, y=y_var, hue=hue_var, log_scale=[10,10], rug=False)
        g = sns.displot(df, x=x_var, y=y_var, row=row_var, col=col_var, hue=hue_var, log_scale=log_scale, rug=False, legend=legend)

    if write_mode:
        basename = "pairs_nulldists"
        outname = f"{basename}.png"
        permtype = '-'.join(list(df["permtype"].unique()))
        outname = outname.replace(f"{basename}",f"pairs_{permtype}-nulldists-extremal")
        if log_scale:
            outname = outname.replace(f"{basename}","{basename}-log")
        for var in ['x', 'y', 'hue', 'row', 'col']:
            varname = eval(f"{var}_var")
            if varname is not None:
                outname = outname.replace(f"{basename}", f"{basename}_{var}-{varname}")
        outpath = os.path.join(outdir, outname)
        futils._write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()
    return g

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
        "-r",
        "--pattern_restriction",
        type=str,
        default=None,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-P",
        "--permtype",
        type=str,
        default=None,
        help="permutation type to restrict to: 'feature' or 'subject' -- forbids combining both nulltypes into single distribution."
    )
    parser.add_argument(
        "-E",
        "--extrema_only",
        default=False,
        action="store_true",
        help="retain only extremal (min/max) valeus of paired null distance distribution (per pair)"
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
    args.dist_vars = dist_vars
    
    df, args = main(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    plot_dists(df, args)
