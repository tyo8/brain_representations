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
import fig_utils as futils
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import fdrcorrection

# global variables 

def_fig_size = (24, 24)
def_label_fontsize = 7 

def_dir_pattern = 'within_*'
def_f_pattern =  '*_vs_*null*'
# exp_outtype="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/null_vs_grad/permtesting/X_grad200_Maps_Psim_dists/data_vs_subjectnull_grad100_Maps_Psim_OR_inner.csv"

modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]
sample_dirnames = {"perm": "permtesting", "bstrap": "subsampling"}

################################################# MAIN FUNCTION ########################################################
########################################################################################################################
def main(args, debug=False):
    if args.fpathlist_path is None: 
        fpath_list = None
    else:
        with open(args.fpathlist_path, 'r') as fin:
            fpath_list = fin.read().split('\n')

    if args.output_dir is None: 
        args.output_dir = os.getcwd()

    if args.pattern_restriction is not None and not args.output_dir.endswith(args.pattern_restriction):
        args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
            print(f"Warning: created new output directory \'{args.output_dir}\'")

    # sample null-data fpath: "data_vs_subjectnull_grad15_Maps_Psim.csv"
    # sample subsample fpath: "bsdists_grad15_Maps_Psim.csv"

    if fpath_list is None and args.input_dir is not None:
        fpath_list = _get_fpath_list(args)

    if args.solo_plots:
        make_solo_plots(fpath_list, dist_type="single", args=args)

    if args.aggregate_plots:
        alldata_list = [ _load(fpath, enforce_match=args.enforce_match) for fpath in fpath_list ]

        null_df = pd.concat(alldata_list, ignore_index=True)
        print(f"total collected dataframe: \n{null_df}")

        for hue_var in ["modality", "feature", "metric", "permtype"]:
            agg_displot(
                    null_df,
                    x_var="Wp_XY",
                    y_var=None,
                    row_var=None,
                    col_var=None,
                    hue_var=hue_var,
                    write_mode=args.write_mode, 
                    outdir=args.output_dir
                    )

    return None

############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty nulled-null distance distribution summaries
########################################################################################################################
def make_solo_plots(fpath_list, dist_type="single", args=None, figs=True, debug=True):

    fpath_groups = list(set([ _get_fpath_types(fpath, dist_type=dist_type)[0] for fpath in fpath_list ]))

    single_df_list = [ pd.concat( 
                                 [pd.read_csv(fpath, index_col=0) for fpath in group if os.path.isfile(fpath)]
                                 ) for group in fpath_groups ]
    for df in single_df_list:
        null_mask = df["datatype"] == "Null"
        if any(null_mask):
            df.loc[null_mask,"datatype"] = df.loc[null_mask].apply( lambda x: "_".join([x["permtype"], x["datatype"]]), axis=1 )

    if debug:
        single_df_list[-1].to_csv("df_tmp.csv")
        with open("fpath_group.txt",'w') as fin:
            for fpath in fpath_groups[-1]:
                fin.write(f"{fpath}\n")

    if figs:
        single_outdirs = [ _get_fpath_types(group[0])[1] for group in fpath_groups ]

        for inputs in list(zip(single_df_list, single_outdirs)):
            if args is None:
                one_displot( inputs[0], outdir=inputs[1], hue_var="datatype" )
            else:
                one_displot(
                        inputs[0],
                        outdir=inputs[1],
                        hue_var="datatype",
                        write_mode = args.write_mode,
                        verbose = args.verbose
                        )
    else:
        return single_df_list

def one_displot(
        df,
        regularize=True,
        log_scale=True,
        x_var="Wp_XY",
        y_var=None,
        hue_var=None,
        row_var=None,
        col_var=None,
        fig_title=None, 
        fig_size=def_fig_size,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):
    if regularize:
        df[x_var] = df[x_var] + 1e-12
        if y_var is not None:
            df[y_var] = df[y_var] + 1e-12

    plot_df = df[ df["datatype"] != "Data" ]
    data_df = df[ df["datatype"] == "Data" ]

    if "Y_name" in df.columns.values:
        name = df["X_name"][0] + "-vs-" + df["Y_name"][0]
    else:
        name = df["X_name"][0]

    if y_var is None:
        g = sns.displot(plot_df, x=x_var, hue=hue_var, multiple="stack", log_scale=args.log_scale, rug=False, element='step')
        # g = sns.displot(df, x=x_var, hue=hue_var, multiple="layer", log_scale=True, rug=False, element='poly')
        # g = sns.displot(df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
        # g = sns.displot(df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
    else:
        g = sns.displot(plot_df, x=x_var, y=y_var, hue=hue_var, log_scale=[10,10], rug=False)
        # g = sns.displot(df, x=x_var, y=y_var, row=row_var, col=col_var, hue=hue_var, log_scale=[10,10], rug=False)

    if not data_df.empty:
        g.refline(x=data_df[x_var], linestyle="--", color="red", label="data distance")

    if write_mode:
        outname = f"{name}_summary.png"
        if log_scale:
            outname = outname.replace("summary","summary-log")
        for var in ['x', 'y', 'hue', 'row', 'col']:
            varname = eval(f"{var}_var")
            if varname is not None:
                outname = outname.replace("summary", f"summary_{var}-{varname}")
        outpath = os.path.join(outdir, outname)
        _write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()


    return None

def agg_displot(
        df,
        regularize=True,
        log_scale=True,
        x_var="Wp_XY",
        y_var="feat_num",
        row_var="modality",
        col_var="feature",
        hue_var="metric",
        fig_title=None, 
        fig_size=def_fig_size,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):

    if regularize:
        df[x_var] = df[x_var] + 1e-12
        if y_var is not None:
            df[y_var] = df[y_var] + 1e-12

    if y_var is None:
        g = sns.displot(df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False, element='step')
        # g = sns.displot(df, x=x_var, hue=hue_var, multiple="layer", log_scale=True, rug=False, element='poly')
        # g = sns.displot(df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
        # g = sns.displot(df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
    else:
        g = sns.displot(df, x=x_var, y=y_var, hue=hue_var, log_scale=[10,10], rug=False)
        # g = sns.displot(df, x=x_var, y=y_var, row=row_var, col=col_var, hue=hue_var, log_scale=[10,10], rug=False)

    if write_mode:
        outname = "nulldists.png"
        if log_scale:
            outname = outname.replace("nulldists","nulldists-log")
        for var in ['x', 'y', 'hue', 'row', 'col']:
            varname = eval(f"{var}_var")
            if varname is not None:
                outname = outname.replace("nulldists", f"nulldists_{var}-{varname}")
        outpath = os.path.join(outdir, outname)
        _write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()


    return g
########################################################################################################################
########################################################################################################################
    

# compute secondary statistics
########################################################################################################################
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def _get_fpath_list(args, debug=True):
    if args.pattern_restriction is None:
        args.pattern_restriction = ""

    subdir_pattern = os.path.join(
            args.dir_pattern, 
            sample_dirnames[args.sample_type],
            f"X_*{args.pattern_restriction}*_dists"
            )
    match_pattern = os.path.join(
            args.input_dir, 
            subdir_pattern, 
            f"{args.f_pattern}.csv"
            )

    fpath_list = glob.glob(match_pattern)
    fpath_list.sort()

    if debug:
        print(f"general match pattern is: \n\'{match_pattern}\'")

    if args.enforce_match:
        print("enforcing modality, feature, and metric matching between data and null")
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries prior to match enforcement.")
        fpath_list = [fpath for fpath in fpath_list if '_'.join(_parse_fpath(fpath, metric=False)) in os.path.basename(fpath)]
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries after match enforcement.")
    return fpath_list


def _load(input_fpath, enforce_match=True, debug=False):
    data_df = pd.read_csv(input_fpath, index_col=0)

    if enforce_match:
        data_modality, data_feature, data_metric = _parse_fpath(input_fpath, metric=True)
        data_df= data_df[data_df["modality"] == data_modality]
        data_df= data_df[data_df["feature"] == data_feature]
        data_df= data_df[data_df["metric"] == data_metric]

    if debug:
        print(f"df before expansion: \n{data_df}")

    if data_df.empty:
        if debug:
            print(f"Loaded empty DataFrame from path: \n{input_fpath}")
        data_df["rank"] = None
        data_df["feat_num"] = None
        return data_df
    else:
        data_df[["modality","rank"]] = data_df.apply( lambda x: _pull_rank(x["modality"]), result_type="expand", axis=1 )
        data_df["feat_num"] = data_df.apply( lambda x: _pull_feat_num(x["rank"], x["feature"]), axis=1 )

    if debug:
        print(f"df after expansion: \n{data_df}")
    return data_df

def _pull_rank(long_method, debug=False):
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
        if debug:
            print(f"[method, rank] = {[method, int(rank)]}")
    return method, int(rank)

def _pull_feat_num(rank, feature):
    if isinstance(rank, float):
        rank = int(10**rank)    # assumes that non-integer 'rank' is actually log10(rank)

    if 'NM' in feature:
        feat_num = rank * (rank - 1) / 2
    elif 'Map' in feature:
        feat_num = rank * 91282
    elif 'Amps' in feature:
        feat_num = rank
    else:
        raise Exception("Unrecognized feature type")
    return int(feat_num)

def _parse_fpath(fpath, metric=True):
    longname = os.path.basename(os.path.dirname(fpath))
    name = longname.replace("_dists","").replace("X_","")
    modality, feature, metric = name.split('_', maxsplit=2)
    if metric:
        return modality, feature, metric
    else:
        return modality, feature

def _get_fpath_types(fpath, dist_type="single"):
    if dist_type == "single":
        dirname = os.path.dirname(fpath)
        if "permtesting" in fpath:
            name = '_'.join(_parse_fpath(fpath))
            basedir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        elif "subsampling" in fpath:
            name = os.path.basename(fpath).replace('.csv','').replace('bsdists_','')
            basedir = os.path.dirname(os.path.dirname(fpath))
        else:
            raise ValueError(f"did not see expected filepath sampling convention name in given fpath: \n{fpath}")

        featnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_featurenull_{name}.csv")
        subjnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_subjectnull_{name}.csv")
        subsamplepath = os.path.join(basedir, "subsampling", f"bsdists_{name}.csv")
        outdir = basedir
    elif dist_type == "pair":
        xname = '_'.join(_parse_fpath(fpath))
        yname = re.split(r'vs.', os.path.basename(fpath))[1].split('_null')[0]
        basedir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))

        featnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-featurePerms.csv")
        subjnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-subjectPerms.csv")
        subsamplepath = os.path.join(basedir, "All_vs_self", f"X_{xname}_dists", f"bspairdists_{xname}-vs-{yname}.csv")
        outdir = os.path.join(basedir, "single-pair_figures")
    else:
        raise ValueError(f"Unrecognized distance output file type \'{dist_type}\'")

    return (featnullspath, subjnullspath, subsamplepath), outdir 





def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):
    fig.set_size_inches(fig_size, forward=False)
    if not os.path.isdir(os.path.basename(outpath)):
        os.mkdir(os.path.basename(outpath))
        Warning(f"Created new output directory: \n{os.path.basename(outpath)}")
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
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="figure output directory"
    )
    parser.add_argument(
        "-S",
        "--solo_plots",
        default=False,
        action="store_true",
        help="flag to perform solo plots"
    )
    parser.add_argument(
        "-A",
        "--aggregate_plots",
        default=False,
        action="store_true",
        help="flag to perform solo plots"
    )
    parser.add_argument(
        "-d",
        "--dir_pattern",
        type=str,
        default=def_dir_pattern,
        help="search pattern to find desired directories"
    )
    parser.add_argument(
        "-t",
        "--sample_type",
        type=str,
        default="perm",
        help="Specify whether sampling randomness comes from bootstrapping or (indexing) permutation: \'perm\' or \'bsdist\'"
    )
    parser.add_argument(
        "-f",
        "--f_pattern",
        type=str,
        default=def_f_pattern,
        help="search pattern to find desired files"
    )
    parser.add_argument(
        "-r",
        "--pattern_restriction",
        type=str,
        default=None,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-z",
        "--enforce_match",
        default=True,
        action="store_false",
        help="turn off modality + feature + metric match enforcement between data and null"
    )
    parser.add_argument(
        "-F",
        "--fpathlist_path",
        type=str,
        default=None,
        help="filepath to .csv (or .txt) list of filepaths to results to be visualized"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    parser.add_argument(
        "-w",
        "--write_mode",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    args = parser.parse_args()

    main(args, debug=False)
    
