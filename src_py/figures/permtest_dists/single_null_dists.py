import re
import os
import ast
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

# global variables 

def_label_fontsize = 7 

def_dir_pattern = 'within_*'
def_f_pattern =  '*_vs_*null*'
# exp_outtype="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/null_vs_grad/permtesting/X_grad200_Maps_Psim_dists/data_vs_subjectnull_grad100_Maps_Psim_OR_inner.csv"

modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]
eps_global = 1e-9

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
        fpath_list = futils._get_fpath_list(args)

    if args.ROC_analysis:
        auc_df = make_AUC_plots(fpath_list, args, debug=debug)
        auc_mask = futils._get_auc_mask(auc_df=auc_df, args=args)
    else:
        auc_mask = futils._get_auc_mask(fpath_list=fpath_list, args=args)

    if args.AUC_filter:
        print(f"Filtering by AUC significance (at alpha={args.alpha})")
        call_mask = lambda x: auc_mask[x]
        fpath_list = [ fpath for fpath in fpath_list if call_mask("_".join(futils._parse_fpath(fpath, pathtype="solo"))) ]
        print(f"Retained {len(fpath_list)} filepaths.")

    if args.solo_plots:
        args.fig_size=(8,8)
        make_solo_plots(fpath_list, dist_type="single", args=args)

    if args.aggregate_plots:
        args.fig_size=(12,12)
        make_agg_plots(fpath_list, args)

    if args.distribution_plots:
        args.fig_size=None
        make_distribution_plots(fpath_list, dist_type="single", args=args, verbose=args.verbose)

    if args.ROC_analysis:
        return auc_df
    else:
        return None

############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty nulled-null distance distribution summaries
########################################################################################################################
def make_solo_plots(fpath_list, dist_type="single", args=None, debug=False):
    
    merged_df_list = futils.merged_dfs(fpath_list, dist_type=dist_type)
    fpath_groups = list(set([ futils._get_fpath_types(fpath, dist_type=dist_type)[0] for fpath in fpath_list ]))

    solo_outdirs = [ futils._get_fpath_types(group[0], dist_type=dist_type)[1] for group in fpath_groups ]
    
    aesthetic_renamer, denamer = futils.get_aesthetic_names(dist_type)

    for x_var in [ aesthetic_renamer[var] for var in ["Wp_XY", "PDY_diag"] ]:
        for (df, outdir) in list(zip(merged_df_list, solo_outdirs)):
            if df.empty:
                continue
            plot_df = df.rename( mapper=aesthetic_renamer, axis=1, inplace=False )
            if args is None:
                one_displot( 
                            plot_df, 
                            x_var=x_var, 
                            denamer=denamer, 
                            outdir=outdir, 
                            hue_var="datatype" 
                            )
            else:
                one_displot(
                        plot_df,
                        x_var=x_var,
                        outdir=outdir,
                        hue_var="datatype",
                        log_scale = args.log_scale,
                        denamer=denamer,
                        write_mode = args.write_mode,
                        fig_size = args.fig_size,
                        verbose = args.verbose
                        )
    return merged_df_list


def make_AUC_plots(fpath_list, args, debug=False):

    df_list = futils.merged_dfs(fpath_list, dist_type="single", verbose=debug, debug=debug)

    if debug:
        ### debugging code ###
        print(f"All datatypes in df_list: {set([df.datatype.unique() for df in df_list])}")
        ### debugging code ###
    outdir = os.path.join(args.output_dir, "ROC_analysis")
    auc_df = fstats.do_ROC_analysis(
                                    df_list,
                                    outdir=outdir,
                                    distvars=["Wp_XY", "PDY_diag"], 
                                    write_mode=args.write_mode
                                    )
    if args.write_mode:
        AUC_posthoc(auc_df, outdir=outdir, alpha=args.alpha)

    return auc_df


def make_agg_plots(fpath_list, args):
    alldata_list = [ futils._load(fpath, load_type="solo", enforce_match=args.enforce_match) for fpath in fpath_list ]

    null_df = pd.concat(alldata_list, ignore_index=True)
    if args.verbose:
        print(f"total collected dataframe (aggregated plots): \n{null_df}")

    if "permtype" in null_df.columns.values:
        if len(null_df["permtype"].unique()) > 1:
            hue_list = ["modality", "feature", "metric", "permtype"]
    else:
        hue_list = ["modality", "feature", "metric"]

    for hue_var in hue_list:
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


def make_distribution_plots(fpath_list, dist_type="single", args=None, debug=False, verbose=True):
    merged_df_list = futils.merged_dfs(fpath_list, dist_type=dist_type)

    alldata_df = pd.concat(merged_df_list, ignore_index=True)

    if verbose:
        print(f"total collected dataframe (distribution plots): \n{alldata_df}")

    aesthetic_renamer, denamer = futils.get_aesthetic_names(dist_type)

    groupings = [None, aesthetic_renamer["modality"], aesthetic_renamer["feature"], aesthetic_renamer["metric"] ]
    hue_vars = ["datatype"]
    if dist_type=="single":
        distvars = [aesthetic_renamer["Wp_XY"], aesthetic_renamer["PDY_diag"]]
    elif dist_type=="pair":
        distvars = [aesthetic_renamer["Wp_XY"]]

    if args is None:
        spec_string = None
    else:
        spec_string = args.pattern_restriction

    for distvar in distvars:
        for x_var in groupings[1:]:
            for hue_var in hue_vars:
                for row_var in groupings:
                    for col_var in groupings:
                        if row_var is not None:
                            if (col_var == row_var) or (x_var == row_var):
                                continue
                        if col_var is not None:
                            if (col_var == row_var) or (x_var == col_var):
                                continue

                        if dist_type=="pair":
                            df = subselect_pairs(
                                    alldata_df.copy(),
                                    x_var=denamer[x_var],
                                    row_var=denamer[row_var],
                                    col_var=denamer[col_var]
                                    )
                        elif dist_type=="single":
                            df = alldata_df.copy()
                        else:
                            raise ValueError(f"Unknown distribution type \'{dist_type}\'")
                        
                        if df.empty:
                            continue

                        if debug:
                            plotvars={"y_var": distvar, "x_var": x_var, "hue_var": hue_var, "row_var": row_var, "col_var": col_var}
                            print(f"plotting distributional category plot with plotvars \n{plotvars} \nof dataframe with columns \n{df.columns.values}")

                        plot_df = df.rename( mapper=aesthetic_renamer , axis=1 )

                        if denamer[distvar] == "Wp_XY":
                            plot_df.drop( index=plot_df[plot_df["datatype"] == "Data"].index, inplace=True )

                        distribution_catplot(
                                plot_df,
                                x_var = x_var,
                                y_var = distvar,
                                hue_var = hue_var,
                                row_var = row_var,
                                col_var = col_var,
                                dist_type = dist_type,
                                denamer = denamer,
                                spec_string = spec_string,
                                write_mode=args.write_mode, 
                                outdir=args.output_dir
                                )
    return None

    
########################################################################################################################
    
def one_displot(
        df,
        x_var="Wp_XY",
        y_var=None,
        hue_var=None,
        row_var=None,
        col_var=None,
        regularize=True,
        log_scale=True,
        epsilon=eps_global,
        fig_title=None, 
        fig_size=None,
        denamer=None,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):
    if regularize:
        df.loc[:,x_var] = df.loc[:,x_var] + epsilon
        if y_var is not None:
            df.loc[:,y_var] = df.loc[:,y_var] + epsilon

    data_df = df[ df["datatype"] == "Data" ].copy()
    plot_df = df[ df["datatype"] != "Data" ]

    if "Y_name" in df.columns.values:
        name = df["X_name"][0] + "-vs-" + df["Y_name"][0]
    else:
        name = df["X_name"][0]

    if y_var is None:
        g = sns.displot(plot_df, x=x_var, hue=hue_var, multiple="layer", log_scale=log_scale, rug=False, element='poly', stat='proportion')
        # g = sns.displot(plot_df, x=x_var, hue=hue_var, multiple="dodge", log_scale=log_scale, rug=False, element='step')
        # g = sns.displot(plot_df, x=x_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
        # g = sns.displot(plot_df, x=x_var, row=row_var, col=col_var, hue=hue_var, multiple="stack", log_scale=True, rug=False)
    else:
        g = sns.displot(plot_df, x=x_var, y=y_var, hue=hue_var, log_scale=[10,10], rug=False)
        # g = sns.displot(df, x=x_var, y=y_var, row=row_var, col=col_var, hue=hue_var, log_scale=[10,10], rug=False)

    if not data_df.empty and (denamer[x_var]=="Wp_XY"):
        dataline = np.unique(data_df[x_var].values.flatten())
        assert len(dataline) == 1, f"found more than one \'{x_var}\' data value in \'{name}\'!"
        add_ref=True
    elif denamer[x_var]=="PDY_diag":
        dataline = np.unique(plot_df["PDX_diag"].values.flatten())
        if len(dataline) == 1:
            Warning(f"found more than one \'{x_var}\' data value in \'{name}\'!")
        add_ref=True
    else:
        add_ref=False
    if add_ref:
        g.refline(x=dataline[0], linestyle="--", color="red", label="Data")


    title = f"Wasserstein distance distributions for \n{name}"
    g.fig.suptitle(title, fontsize='x-large')

    if write_mode:
        basename = "summary"
        outname = f"{name}_{basename}.png"
        if log_scale:
            outname = outname.replace(basename,f"{basename}-log")
        for var in ['row', 'col', 'hue', 'x', 'y']:
            varname = eval(f"{var}_var")
            if varname is not None:
                if (denamer is not None) and (varname in denamer.keys()):
                    outname = outname.replace(basename, f"{basename}_{var}-{denamer[varname]}")
                else:
                    outname = outname.replace(basename, f"{basename}_{var}-{varname}")
        if log_scale:
            outname = outname.replace(basename,f"{basename}-log")
        outpath = os.path.join(outdir, outname)
        futils._write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        g.fig.set_size_inches(fig_size, forward=True)
        plt.show()


    return None

    
########################################################################################################################
    
def agg_displot(
        df,
        x_var="Wp_XY",
        y_var="feat_num",
        row_var="modality",
        col_var="feature",
        hue_var="metric",
        regularize=True,
        log_scale=True,
        epsilon=eps_global,
        fig_title=None, 
        fig_size=None,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):

    if regularize:
        df.loc[:,x_var] = df.loc[:,x_var] + epsilon
        if y_var is not None:
            df.loc[:,y_var] = df.loc[:,y_var] + epsilon

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
        permtype = '-'.join(list(df["permtype"].unique()))
        outname = outname.replace(f"nulldists",f"{permtype}-nulldists")
        if log_scale:
            outname = outname.replace("nulldists","nulldists-log")
        for var in ['x', 'y', 'hue', 'row', 'col']:
            varname = eval(f"{var}_var")
            if varname is not None:
                outname = outname.replace("nulldists", f"nulldists_{var}-{varname}")
        outpath = os.path.join(outdir, outname)
        futils._write_img(g.fig, outpath, fig_size=fig_size)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()


    return g
    
    
########################################################################################################################
    

def distribution_catplot(
        df,
        x_var=None,
        y_var=None,
        row_var=None,
        col_var=None,
        hue_var="datatype",
        kind = "boxen",
        dist_type="single",
        regularize=True,
        log_scale=True,
        epsilon=eps_global,
        fig_title=None, 
        fig_size=None,
        spec_string=None,
        denamer=None,
        write_mode=True,
        outdir=os.getcwd(),
        verbose=True, 
        debug=False
        ):

    if regularize:
        df.loc[:,y_var] = df.loc[:,y_var] + epsilon

    if spec_string is not None:
        dist_type = f"{dist_type}-{spec_string}"
        if dist_type.endswith('-'):
            dist_type = dist_type[:-1]

    try:
        g = sns.catplot(
                df, 
                y=y_var,
                x=x_var, 
                hue=hue_var,
                row=row_var,
                col=col_var,
                kind=kind,
                log_scale=log_scale, 
                fill=True             # a violin/boxen plot option
                )
    except ValueError as err:
        print(f"Failed with error: {err}")
        print(f"offending dataframe has column set: \n{df.columns.values} \nand values: \n{df}")
        exit()

    title = f"Wasserstein distance distributions\ngrouped by {x_var}"
#   for var in ['row', 'col', 'hue']:
#       varname = eval(f"{var}_var")
#       if varname is not None:
#           title = title + f"\nand {varname}"
    g.fig.suptitle(title, fontsize='x-large', y=1)
    g.fig.subplots_adjust(top=0.85)
    

    if write_mode:
        # kindstring = '-'.join(kinds)
        basename = f"catplot_{dist_type}"
        # outname = f"{kindstring}-{basename}.png"
        outname = f"{kind}-{basename}.png"
        for var in ['row', 'col', 'hue', 'x', 'y']:
            varname = eval(f"{var}_var")
            if varname is not None:
                if (denamer is not None) and (varname in denamer.keys()):
                    outname = outname.replace(basename, f"{basename}_{var}-{denamer[varname]}")
                else:
                    outname = outname.replace(basename, f"{basename}_{var}-{varname}")
        if log_scale:
            outname = outname.replace(basename,f"{basename}-log")
        outpath = os.path.join(outdir, outname)
        futils._write_img(g.fig, outpath, fig_size=None)   # use 'fig_size=None' to allow auto-selection of aspect ratio
        plt.close()
    else:
        plt.show()


    return g
########################################################################################################################
########################################################################################################################
    

# AUC_plotting functions
########################################################################################################################
# post-hoc analysis of aggregated AUC data
def AUC_posthoc(df, outdir=None, alpha=0.01, epsilon=eps_global, x_var="reg_overlap", row_var="permtype", col_var="ROC_variable"):
    if x_var not in df.columns.values:
        df[x_var] = df["overlap"] + epsilon
    if outdir is None:
        outdir = os.getcwd()

    AUC_displot(df, outdir, alpha=alpha, x_var=x_var, row_var=row_var, col_var=col_var)
    sig_relplot(df, outdir, alpha=alpha, x_var=x_var, hue_var=row_var, style_var=col_var)
    return None

def AUC_displot(df, outdir, alpha=0.01, x_var="reg_overlap", row_var="permtype", col_var="ROC_variable"):
    g = sns.displot(
            data = df, 
            x=x_var, col=col_var, row=row_var, 
            log_scale=True)

    g.refline(x=alpha, color='r', linestyle='--')
    g.fig.suptitle("Distributions of AUC over Brain Reps", fontsize='x-large', y=1)
    g.fig.subplots_adjust(top=0.95)

    outname = f"AUC_histograms_alpha{alpha}.png".replace("0.","")
    outpath = os.path.join(outdir, outname)
    futils._write_img(g.fig, outpath, fig_size=None)
    return None

def sig_relplot(df, outdir, alpha=0.01, x_var="reg_overlap", hue_var="permtype", style_var="ROC_variable"):
    sig_list = []
    thresh_range = np.logspace(-6, 0, 1000)
    t_var = "log-alpha(t)"
    y_var = "N_significant"
    style_renamer = {"Wp_XY": "dist. to PD(X)", "PDY_diag": "dist. to emtpy dgm"}
    for i in df[hue_var].unique():
        for j in df[style_var].unique():
                submask = (df[hue_var]==i) & (df[style_var]==j)
                p = [sum(df.loc[submask, x_var] < t) for t in thresh_range]
                sig_list.append( {"subset": f"{i}-{j}", t_var: np.log10(thresh_range), y_var: p, hue_var: i, style_var: style_renamer[j]} )
                asig = df.loc[submask, x_var] < alpha
                print(f"at alpha={alpha}, {sum(asig)} of {len(asig)} {i}-{j} entries are significant.")

    sig_df = pd.concat( [ pd.DataFrame(i) for i in sig_list ], ignore_index=True )

    g = sns.relplot(sig_df, x=t_var, y=y_var, hue=hue_var, style=style_var, kind="line")
    g.refline(x=np.log10(alpha), color='r', linestyle=':')
    g.fig.suptitle("Number of signifcantly non-noise brain representations", fontsize='x-large', y=1)
    g.fig.subplots_adjust(top=0.95)

    outname = f"significance_counts_alpha{alpha}.png".replace("0.","")
    outpath = os.path.join(outdir, outname)
    futils._write_img(g.fig, outpath, fig_size=None)
    return None
########################################################################################################################


# Data wrangling helper functions
########################################################################################################################
def subselect_pairs(df, x_var="modality", denamer=None, row_var=None, col_var=None, debug=True, verbose=False):
    if verbose:
        inputs = {
                "x_var": x_var,
                "row_var": row_var,
                "col_var": col_var
                }
        print(f"subselecting pairs according to: \n{inputs}")
    if debug:
        ### debugging code ###
        print(f"dataframe has pre-subselection columns: \n{df.columns.values}")
        print(f"with pre-subselection values: \n{df}")
        ### debugging code ###

    if (x_var, row_var, col_var) == (None, None, None):
        return df

    if (x_var is not None) and (x_var not in df.columns.values):
        x_mask = df[f"X_{x_var}"] == df[f"Y_{x_var}"]
        df_mask = x_mask
        dropper = [f"Y_{x_var}"]
        renamer = {f"X_{x_var}": x_var}

    if (row_var is not None) and (row_var not in df.columns.values):
        rowvar_mask = df[f"X_{row_var}"] == df[f"Y_{row_var}"]
        df_mask = df_mask & rowvar_mask
        dropper.append( f"Y_{row_var}" )
        renamer = renamer | {f"X_{row_var}": row_var}

    if (col_var is not None) and (col_var not in df.columns.values):
        colvar_mask = df[f"X_{col_var}"] == df[f"Y_{col_var}"]
        df_mask = df_mask & colvar_mask
        dropper.append( f"Y_{col_var}" )
        renamer = renamer | {f"X_{col_var}": col_var}

    if verbose:
        print(f"\ndropping columns {dropper} \nrenaming columns as: {renamer}")

    df.drop( columns=dropper, inplace=True )
    df.rename( columns=renamer, inplace=True )

    df = df.loc[df_mask,:]

    if debug:
        ### debugging code ###
        print(f"dataframe has post-subselection columns: \n{df.columns.values}")
        print(f"with post-subselection values: \n{df}")
        ### debugging code ###

    return df



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
        "-R",
        "--ROC_analysis",
        default=False,
        action="store_true",
        help="flag to perform ROC analysis"
    )
    parser.add_argument(
        "-A",
        "--aggregate_plots",
        default=False,
        action="store_true",
        help="flag to perform aggregate plots"
    )
    parser.add_argument(
        "-D",
        "--distribution_plots",
        default=False,
        action="store_true",
        help="flag to perform distribution plots"
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
        "-Z",
        "--AUC_filter",
        default=True,
        action="store_false",
        help="filter to retain only significantly non-noise brain reps"
    )
    parser.add_argument(
        "-F",
        "--fpathlist_path",
        type=str,
        default=None,
        help="filepath to .csv (or .txt) list of filepaths to results to be visualized"
    )
    parser.add_argument(
        "-L",
        "--log_scale",
        default=False,
        action="store_true",
        help="apply log10 to display values (collapse difference)"
    )
    parser.add_argument(
        "-a",
        "--alpha",
        default=0.05,
        type=float,
        help="significance threshold"
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
    
