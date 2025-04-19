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
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

# global variables 

def_fig_size = (24, 24)
def_label_fontsize = 7 

def_dir_pattern = 'within_*'
def_f_pattern =  '*_vs_*null*'
# exp_outtype="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/null_vs_grad/permtesting/X_grad200_Maps_Psim_dists/data_vs_subjectnull_grad100_Maps_Psim_OR_inner.csv"

modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]
sample_dirnames = {"perm": "permtesting", "bstrap": "subsampling"}
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
        fpath_list = _get_fpath_list(args)


    if args.ROC_analysis:
        auc_df = make_AUC_plots(fpath_list, args, debug=debug)

    if args.solo_plots:
        args.fig_size=(8,8)
        make_solo_plots(fpath_list, dist_type="single", args=args)

    if args.aggregate_plots:
        args.fig_size=(12,12)
        make_agg_plots(fpath_list, args)

    if args.distribution_plots:
        args.fig_size=(6,12)
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
    
    merged_df_list = merged_dfs(fpath_list, dist_type=dist_type)
    fpath_groups = list(set([ _get_fpath_types(fpath, dist_type=dist_type)[0] for fpath in fpath_list ]))

    solo_outdirs = [ _get_fpath_types(group[0], dist_type=dist_type)[1] for group in fpath_groups ]
    
    aesthetic_renamer, denamer = get_better_names(dist_type)

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

    df_list = merged_dfs(fpath_list, dist_type="single", verbose=debug, debug=debug)

    if debug:
        ### debugging code ###
        print(f"All datatypes in df_list: {set([tuple(set(df.datatype)) for df in df_list])}")
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
    alldata_list = [ _load(fpath, enforce_match=args.enforce_match) for fpath in fpath_list ]

    null_df = pd.concat(alldata_list, ignore_index=True)
    if args.verbose:
        print(f"total collected dataframe (aggregated plots): \n{null_df}")

    if "permtype" in null_df.columns.values:
        if len(set(null_df["permtype"])) > 1:
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
    merged_df_list = merged_dfs(fpath_list, dist_type=dist_type)

    alldata_df = pd.concat(merged_df_list, ignore_index=True)

    if verbose:
        print(f"total collected dataframe (distribution plots): \n{alldata_df}")

    aesthetic_renamer, denamer = get_better_names(dist_type)

    groupings = [None, aesthetic_renamer["modality"], aesthetic_renamer["feature"], aesthetic_renamer["metric"] ]
#    if "permtype" in alldata_df.columns.values:
#        hue_vars = ["datatype", "permtype"]
#    else:
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
        fig_size=def_fig_size,
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
        _write_img(g.fig, outpath, fig_size=fig_size)
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
        fig_size=def_fig_size,
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
        fig_size=def_fig_size,
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
        _write_img(g.fig, outpath, fig_size=None)   # use 'fig_size=None' to allow auto-selection of aspect ratio
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
    _write_img(g.fig, outpath, fig_size=None)
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
    _write_img(g.fig, outpath, fig_size=None)
    return None
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def _get_fpath_list(args, debug=False):
    if args.pattern_restriction is None:
        args.pattern_restriction = ""

    basedir_pattern = os.path.join(
            args.input_dir,
            args.dir_pattern, 
            sample_dirnames[args.sample_type]
            )

    if args.sample_type=="perm":
        path_ext = os.path.join(f"X_*{args.pattern_restriction}*_dists", f"{args.f_pattern}.csv")
    elif args.sample_type == "bstrap":
        path_ext = f"bsdists_*{args.pattern_restriction}*.csv"

    match_pattern = os.path.join(
            basedir_pattern,
            path_ext
            )

    fpath_list = glob.glob(match_pattern)
    fpath_list.sort()

    if args.verbose:
        print(f"general match pattern is: \n\'{match_pattern}\'")

    if args.enforce_match and args.verbose:
        print("enforcing modality, feature, and metric matching between data and null")
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries prior to match enforcement.")
        fpath_list = [fpath for fpath in fpath_list if '_'.join(_parse_fpath(fpath, metric=False)) in os.path.basename(fpath)]
        if debug or args.verbose:
            print(f"fpath_list has {len(fpath_list)} entries after match enforcement.")
    return fpath_list


def _load(input_fpath, enforce_match=True, debug=False):
    data_df = pd.read_csv(input_fpath, index_col=0)
    if data_df.empty:
        print(f"pulled empty dataframe from path: \n{input_fpath}")

    data_df = _unify_df(data_df, fpath=input_fpath)

    modality, feature, metric = _parse_fpath(input_fpath, metric=True)
    pars = {
            "modality": modality,
            "feature": feature,
            "metric": metric
            }
    if enforce_match:
        for keyname in ["modality", "feature", "metric"]:
            data_df= data_df[ data_df[keyname] == pars[keyname] ]

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
    if "subsampling" in fpath:
        longname = os.path.basename(fpath).split('.')[0].replace("bsdists_","")
    else:
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
            raise ValueError(f"did not see expected filepath sampling convention name for distance type \'{dist_type}\' in given fpath: \n{fpath}")

        featnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_featurenull_{name}.csv")
        subjnullspath = os.path.join(basedir, "permtesting", f"X_{name}_dists", f"data_vs_subjectnull_{name}.csv")
        subsamplepath = os.path.join(basedir, "subsampling", f"bsdists_{name}.csv")
        outdir = basedir
    elif dist_type == "pair":
        xname = '_'.join(_parse_fpath(fpath))
        yname = re.split(r'vs.', os.path.basename(fpath))[1].split('_null')[0]
        basedir = os.path.dirname(os.path.dirname(os.path.dirname(fpath)))
        name = f"{xname}_vs_{yname}"

        featnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-featurePerms.csv")
        subjnullspath = os.path.join(basedir, "All_vs_AllNull", f"X_{xname}_dists", f"{xname}_vs_{yname}_null-subjectPerms.csv")
        subsamplepath = os.path.join(basedir, "All_vs_self", f"X_{xname}_dists", f"bspairdists_{xname}-vs-{yname}.csv")
        outdir = os.path.join(basedir, "single-pair_figures")
    else:
        raise ValueError(f"Unrecognized distance output file type \'{dist_type}\'")

    return (featnullspath, subjnullspath, subsamplepath), outdir, name


def merged_dfs(fpath_list, dist_type="single", debug=False, verbose=False):
    fpath_groups = list(set([ _get_fpath_types(fpath, dist_type=dist_type)[0] for fpath in fpath_list ]))
    merged_df_list = [ get_merge_df(group) for group in fpath_groups ]

    if verbose:
        if dist_type=="single":
            print(f"evaluating {len(merged_df_list)} merged dataframes.")
            for i,df in enumerate(merged_df_list):
                name = os.path.basename(fpath_groups[i][-1]).replace("bsdists_","").replace(".csv","")
                if debug:
                    datamask = (df.datatype=="Subsamp") | (df.datatype=="Data")
                    df_data = df.loc[datamask,:]
                    if len(df_data) < 1000:
                        print(f"found incomplete or empty subsampling/data (shape={df_data.shape}) at path: \n{fpath_groups[i][-1]}")
                        _ = _get_orig_bars(fpath_groups[i][-1], homdim=1)
                    else:
                        print(f"subsampling data (shape={df_data.shape}) passes inspection -- found at path: \n{fpath_groups[i][-1]}")
                print(f"merged_df from \'{name}\' paths has shape {df.shape}\n")

    if debug:
        ### debugging code ###
        merged_df_list[-1].to_csv("df_tmp.csv")       
        with open("fpath_group.txt",'w') as fin:
            for fpath in fpath_groups[-1]:
                fin.write(f"{fpath}\n")
        ### debugging code ###
    return merged_df_list

def get_merge_df(fpath_group):
    df_list = [pd.read_csv(fpath, index_col=0) for fpath in fpath_group if os.path.isfile(fpath)]

    for df in df_list:
        if df.empty:
            continue
        else:
            df = _unify_df(df)

    if all([df.empty for df in df_list]):
        print("\nCompletely empty dataframe output corresponding to fpath group:")
        for fpath in fpath_group:
            print(f"\t{fpath}")
        merge_df = pd.DataFrame(None)
    else:
        merge_df = pd.concat( df_list, ignore_index=True )

    return merge_df


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


def _unify_df(df, fpath=None):

    if df.empty:
        return df

    # I'm never changing a naming convention again, even if it's terrible; this has been so annoying to deal with
    cols = [ col for col in df.columns.values if ("metric" in col) or ("name" in col) ]
    df[cols] = df[cols].applymap( lambda x: x.replace("Psim_ztrans", "Psim-ztrans") )

    if "X_type" in df.columns.values:
        df.rename( mapper={"X_type":"X_name"}, axis=1, inplace=True )
        if "Y_type" in df.columns.values:
            df.rename( mapper={"Y_type":"Y_name"}, axis=1, inplace=True )
            df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
            df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)

    if not "Y_name" in df.columns.values:
        if "X_name" in df.columns.values:
            df[["modality","feature","metric"]] = df["X_name"].str.split('_', n=2, expand=True)
        else:
            df["X_name"] = df.apply( lambda x: "_".join([x["modality"], x["feature"], x["metric"]]), axis=1 )

        if fpath is not None:
            modality, feature, metric = _parse_fpath(fpath, metric=True)
            pars = {
                    "modality": modality,
                    "feature": feature,
                    "metric": metric
                    }
            for keyname in ["modality", "feature", "metric"]:
                if keyname not in df.columns.values:
                    df[keyname] = pars[keyname]
                else:
                    assert all(df[keyname] == pars[keyname]), f"Data value and filename conflict for key \'{keyname}\' in file: \n\'{fpath}\'"

        df[["modality","rank"]] = df.apply( lambda x: _pull_rank(x["modality"]), result_type="expand", axis=1 )
        df["feat_num"] = df.apply( lambda x: _pull_feat_num(x["rank"], x["feature"]), axis=1 )
    else:
        df[["X_modality","X_feature","X_metric"]] = df["X_name"].str.split('_', n=2, expand=True)
        df[["Y_modality","Y_feature","Y_metric"]] = df["Y_name"].str.split('_', n=2, expand=True)


    if "permtype" in df.columns.values:
        try:
            null_mask = df["datatype"]=="Null" 
            df.loc[null_mask, "datatype"] = df.loc[null_mask,:].apply( lambda x: "_".join([str(x["permtype"]), str(x["datatype"])]), axis=1 )
        except TypeError:
            print(f"encountered unexpected float values in \'permtype\' field: {list(set(df.permtype))}")
            print("attempting to resolve by forcing type to \'str\'.")

    if "permlabel" in df.columns.values:
        df.drop( labels = ["permlabel"], axis=1, inplace=True )

    if "taglist" in df.columns.values:
        df.drop( labels = ["taglist", "X_path"], axis=1, inplace=True )
        if "Wp_XY" not in df.columns.values:
            renamer={"Wp_XXhat_i":"Wp_XY", "PDXhat_diag_i":"PDY_diag"}
            df = _add_data_slice(df)
        else:
            renamer={"Wp_XhatYhat_i":"Wp_XY"}
            to_drop = ["Wp_XY", "Y_path", "Wp_YYhat_i", "Wp_XXhat_i", "dIM_YYhat_i", "PDXhat_diag_i", "PDYhat_diag_i",
                    "dIM_XXhat_i", "Mean Wp Approximation Difference", "Wphat_XY", "Wphat0_XY_i"]
            df.drop( labels = to_drop, axis=1, inplace=True )
        df.rename( mapper=renamer, axis=1, inplace=True )

    return df

def _get_orig_bars(
        bsdist_fpath, 
        homdim=1, 
        basedir="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/full-scale-expmt"
        ):
    xname = os.path.basename(bsdist_fpath).replace("bsdists","phom_data").replace(".csv","_dists")
    searchdir = os.path.join(basedir, "within*", "*_*", xname)
    
    [bars_fpath] = glob.glob(os.path.join(searchdir, "phom_X.txt"))     # should have exactly 1 result
    barlines = []
    with open(bars_fpath, 'r') as fin:
        phom = fin.read().split('\n')

    idx = phom.index("persistent homology intervals in dim 1:")
    barlines = phom[idx+1:-1]           # assumes all files have \n or EOF char as last line
    print(f"Found {int(len(barlines)/3)} H1 bars in original data at path: \n{bars_fpath}:")
    if len(barlines) > 0:
        print("with values:")
        for line in barlines:
            print(line)
        if len(glob.glob(os.path.join(searchdir, "B1match_dim1_n1000.txt"))) > 0:     # should have <=1 result
            [b1match_fpath] = glob.glob(os.path.join(searchdir, "B1match_dim1_n1000.txt"))     # should have <=1 result
            b1match_vec = np.loadtxt(b1match_fpath)
            print(f"Of {len(b1match_vec)} recorded attempted matches, {np.count_nonzero(b1match_vec)} produced matches with nonzero affinity. ref:\n{b1match_fpath}")
        else:
            print("-----No corresponding B1match file found!-----")
    return None

def _add_data_slice(df):
    df.loc[-1] = df.loc[0].copy()
    df.loc[-1, "datatype"] = "Data"
    df.loc[-1, "Wp_XXhat_i"] = 0
    df.loc[-1, "PDXhat_diag_i"] = df.loc[-1, "PDX_diag"]
    return df

def get_better_names(dist_type):
    if dist_type=="single":
        better_Wp_name = "Wasserstein distance (full vs. null/subsamp)"
    elif dist_type=="pair":
        better_Wp_name = "Wasserstein distance (paired full/null/subsamp BRs)"

    renamer = {
            "Wp_XY": better_Wp_name, 
            "PDY_diag": "Wasserstein distance from empty diagram",
            "modality": "Brain parcellation", 
            "feature": "Feature",
            "metric": "Metric"
            }
    denamer = {v: k for k, v in renamer.items()}      # inverse dictionary of 'renamer'
    return renamer, denamer


def _write_list(outpath, list_out):
    with open(outpath, 'w') as fout:
        fout.write(list_out.__str__())

def _write_img(fig, outpath, fig_size=def_fig_size):
    if fig_size is not None:
        fig.set_size_inches(fig_size, forward=False)
    if not os.path.isdir(os.path.dirname(outpath)):
        os.mkdir(os.path.dirname(outpath))
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
        default=0.01,
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
    
