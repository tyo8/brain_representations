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

# dist_vars = ["Wp_XYNull_min", "Wp_XYNull_max", "Wp_XYNull_mean", "Wp_XYNull_std", "PDX_diag", "PDY_diag"]
dist_vars = ["Wp_XYNull_min", "Wp_XYNull_max", "Wp_XYNull_mean", "Wp_XYNull_std"]
def_fig_size = (24, 24)

def main(args, debug=False):
    if args.pattern_restriction is not None:
        if not args.output_dir.endswith(args.pattern_restriction):
            args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
        args.dir_pattern=f'*X_*{args.pattern_restriction}*_dists'
        args.f_pattern = f'*{args.pattern_restriction}*_vs_*{args.pattern_restriction}*'
    else:
        args.dir_pattern=f'X_*_dists'
        args.f_pattern = f'*_vs_*'

    if args.verbose:
        var_dict = vars(args)
        for varname in list(var_dict.keys()):
            print(f"The argument \'{varname}\' has been initialized with value: {var_dict[varname]}")

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {args.output_dir}")
        os.mkdir(args.output_dir)

    if args.fpathlist_path is None: 
        search_pattern = os.path.join(args.input_dir, args.dir_pattern, args.f_pattern)
        fpath_list = glob.glob(search_pattern)
    else:
        with open(args.fpathlist_path, 'r') as fin:
            fpath_list = fin.read().split('\n')

    df = pd.concat([ _load(fpath, extrema_only=args.extrema_only) for fpath in fpath_list ], ignore_index=True)

    if args.verbose:
        print(f"collected summary dataframe: \n{df}")
        for colname in df.columns.values:
            print(f"Number of unique values in column \'{colname}\' is {len(set(df[colname].values))}")

    if args.extrema_only:
        df = df.filter(["Wp_XYNull_min", "Wp_XYNull_max"], axis=1)

    return df, args

def plot_dists(df, args):
    one_displot(
            df,
            log_scale=args.log_scale,
            extrema_only=args.extrema_only,
            write_mode=args.write_mode, 
            outdir=args.output_dir
            )
    if not args.extrema_only:
        for dist_var in dist_vars:
            for var in ["modality", "feature", "metric"]:
                df[var] = df.apply(lambda x: frozenset([x[f"X_{var}"], x[f"Y_{var}"]]), axis=1)
                row_var = f"X_{var}"
                col_var = f"Y_{var}"
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
        fig_size=def_fig_size,
        extrema_only=False,
        legend=True,
        log_scale=True,
        regularize=True,
        epsilon=1e-12,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):

    if x_var not in df.columns.values and x_var is not None:
        print(f"distribution variable {x_var} not in filtered dataframe. Skipping.")
        return None

    if log_scale and regularize:
        if x_var is None:
            for name in df.columns.values:
                df[df[name] == 0] = epsilon 
        else:
            df[x_var] = df[x_var] + epsilon
        if y_var is not None:
            if isinstance(df[y_var].values, np.ndarray):
                log_scale = [10, 10]
                df[y_var] = df[y_var] + epsilon
            else:
                y_num = len(set(df[y_var]))
                legend = y_num <= 100
                log_scale = [10, None]

    if hue_var is not None:
        hue_num = len(set(df[hue_var]))
        legend = hue_num <= 20

    if y_var is None:
        if x_var is None:
            if extrema_only:
                plot_df = df.filter(["Wp_XYNull_min", "Wp_XYNull_max"], axis=1)
                element = 'bars'; multiple = 'dodge'
            else:
                plot_df = df.filter(dist_vars, axis=1)
                element = 'poly'; multiple = 'layer'
            if verbose:
                print(f"wide-form plotting histogram of dataframe df=\n{plot_df}")
            g = sns.displot(data=plot_df, multiple=multiple, log_scale=log_scale, rug=False, legend=legend, element=element)
        else:
            # g = sns.displot(df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
            g = sns.displot(df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="stack", log_scale=log_scale, rug=False, legend=legend, element='poly')
    else:
        # g = sns.displot(df, x=x_var, y=y_var, hue=hue_var, log_scale=[10,10], rug=False)
        g = sns.displot(df, x=x_var, y=y_var, row=row_var, col=col_var, hue=hue_var, log_scale=log_scale, rug=False, legend=legend)

    if write_mode:
        outname = "pairs_nulldists.png"
        if log_scale:
            outname = outname.replace("pairs_nulldists","pairs_nulldists-log")
        if extrema_only:
            outname = outname.replace("pairs_nulldists","pairs_nulldists-extremal")
        for var in ['x', 'y', 'hue', 'row', 'col']:
            varname = eval(f"{var}_var")
            if varname is not None:
                outname = outname.replace("pairs_nulldists", f"pairs_nulldists_{var}-{varname}")
        outpath = os.path.join(outdir, outname)
        _write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()
    return g

########################################################################################################################

def pull_data(
        fpath_list=None, 
        extrema_only=True, 
        debug=False
        ):

    df_list = [ _load(fpath, extrema_only=extrema_only) for fpath in fpath_list ]

    return df_list


def _load(input_fpath, extrema_only=False, parse_longname=True):
    with open(input_fpath, 'r') as fin:
        data_df = pd.DataFrame(json.load(fin))

    if parse_longname:
        data_df[["X_modality","X_feature","X_metric"]] = data_df["X_type"].str.split('_', n=2, expand=True)
        data_df[["Y_modality","Y_feature","Y_metric"]] = data_df["Y_type"].str.split('_', n=2, expand=True)
        # data_df["XY_type"] = data_df.apply(lambda x: "_vs_".join([x["X_type"], x["Y_type"]]), axis=1)

    null_df = data_df[data_df["datatype"] == "Null"]
    data_df = data_df[data_df["datatype"] == "Data"]
    data_df["Wp_XYNull_min"] = np.min(null_df["Wp_XY"])
    data_df["Wp_XYNull_max"] = np.max(null_df["Wp_XY"])

    if not extrema_only:
        data_df["Wp_XYNull_mean"] = np.mean(null_df["Wp_XY"])
        data_df["Wp_XYNull_std"] = np.std(null_df["Wp_XY"])
        
    data_df.drop(["X_type","Y_type", "Wp_XY", "permtype", "permlabel", "datatype"], axis=1, inplace=True)

    return data_df


def _write_img(fig, outpath, fig_size=def_fig_size):
    fig.set_size_inches(fig_size, forward=False)
    fig.savefig(outpath, dpi=600)
    print(f"saved to {outpath}")
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
    
    df, args = main(args)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    plot_dists(df, args)
