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

def_scatter_vars = ["Wp_XY", "PDX_diag", "PDY_diag"]
def_pattern='within_*/permtesting/X_*_dists', 

# exp_outtype="/home/tyo/Documents/Personomics_Lab/Experiments/brain_representations/phom_analysis/stability_distances/exp_results/null_vs_grad/permtesting/X_grad200_Maps_Psim_dists/data_vs_subjectnull_grad100_Maps_Psim_OR_inner.csv"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

# potential separating vars = ["modality", "dimension", "feature", "metric", "permtype"]
#sample entry:
#   {
#       "modality": "grad25",
#       "feature": "Maps",
#       "metric": "inner",
#       "datatype": "Null",
#       "permtype": "subject",
#       "permlabel": "perm_set0_n64620",
#       "Wp_XY": 0.00374385142019842,
#       "PDX_diag": 0.04047972885902275,
#       "PDY_diag": 0.005473396660210167
#   },
#
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

    alldata_list = pull_data(
            fpath_list = fpath_list,
            parent_dir = args.input_dir, 
            dir_pattern=f'within_*/permtesting/X_*{args.pattern_restriction}*_dists',
            f_pattern = '*_vs_*null*'
            )

    null_df = pd.concat(alldata_list, ignore_index=True)
    print(f"total collected dataframe: \n{null_df}")

#             for y_var in ["feat_num", "PDY_diag", None]:
#                 if not x_var==y_var:
    for hue_var in ["modality", "feature", "metric", "permtype"]:
        one_displot(
                null_df,
                x_var="Wp_XY",
                y_var=None,
                row_var=None,
                col_var=None,
                hue_var=hue_var,
                write_mode=args.write_mode, 
                outdir=args.output_dir
                )

#   generate_scatter_plots(
#           args.output_dir, 
#           null_df, 
#           hue_var=hue_var, 
#           style_var=style_var, 
#           write_mode=args.write_mode
#           )
    return None

############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty nulled-null distance distribution summaries
########################################################################################################################
def one_displot(
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
# handle scatter plots
########################################################################################################################
#ef generate_scatter_plots(
#       output_dir,
#       null_df,
#       hue_var="modality",
#       size_var="metric",
#       style_var="feature",
#       write_mode=True,
#       debug=True
#       ):
#
#   all_plotvars = [colname for colname in null_df.columns if colname.endswith("_i")]
#   var_pairs = list(itertools.combinations(all_plotvars, 2))
#   var_triples = []
#   # var_triples = list(itertools.combinations(all_plotvars, 3))
#
#   df_list = [df for _, df in null_df.groupby(null_df['Name'])]
#   
#   if debug:
#       ### debugging code ###
#       print("Generating scater plots...\nSpecified inputs:")
#       print("output_directory:\n", output_dir)
#       print("dataframe (columns):\n", null_df.columns)
#       print("hue_var:\n", hue_var)
#       print("style_var:\n", style_var)
#       print(f"input dataframe split into {len(df_list)} subframes:", [list(df["Name"])[0] for df in df_list])
#       print("set of plotting variable pairs:\n", var_pairs)
#       print("set of plotting variable triples:\n", var_triples)
#       ### debugging code ###
#
#
#   for df in df_list:
#       name = list(df["Name"])[0]
#       for pair in var_pairs:
#           scatter_plot(
#                   df, pair[0], pair[1], 
#                   plt_title = _construct_title((name, pair[0], pair[1]), title_type = "scatter"), 
#                   outdir = output_dir,
#                   style_var = style_var, 
#                   hue_var = hue_var,
#                   space_name = name,
#                   write_mode = write_mode
#                   )
#       for triple in var_triples:
#           scatter_plot(
#                   df, triple[0], triple[1], z_var=triple[2],
#                   plt_title = _construct_title((name, triple[0], triple[1], triple[2]), title_type = "scatter"), 
#                   outdir = output_dir,
#                   style_var = style_var, 
#                   hue_var = hue_var,
#                   space_name = name,
#                   write_mode = write_mode
#                   )
#
#
#
#
# Figure plotting functions
#ef scatter_plot(
#       dataframe, x_var, y_var, z_var=None, space_name=None,
#       write_mode=True, plt_title = None, outdir=None,
#       style_var="noise_lvl", style_order=None,
#       hue_var="emb_dim", hue_order=None
#       ):
#
#   if z_var is None:
#       fig, ax = plt.subplots()
#       g = sns.scatterplot(
#               data = dataframe,
#               x = x_var,
#               y = y_var,
#               markers = True,
#               style = style_var,
#               style_order = style_order,
#               hue = hue_var,
#               hue_order = hue_order,
#               legend = "brief"
#               )
#       g.set(xlabel = x_var)
#       g.set(ylabel = y_var)
#       g.set(title = plt_title)
#   elif isinstance(z_var, str):
#       cmap = matplotlib.colors.ListedColormap(sns.color_palette("Spectral", 256).as_hex())    
#       fig, ax = plt.subplots()
#       ax = fig.add_subplot(projection = '3d')
#       mdict = _manual_stylemap(dataframe[style_var].drop_duplicates().sort_values(), style_order=style_order)
#       cdict = _manual_colormap(dataframe[hue_var].drop_duplicates().sort_values())
#       sc = ax.scatter(
#               dataframe[x_var],
#               dataframe[y_var],
#               dataframe[z_var],
#               marker = list(map(mdict.get, list(dataframe[style_var]))),
#               c = list(map(cdict.get, list(dataframe[hue_var]))),
#               cmap = cmap
#               )
#       ax.set_xlabel( x_var )
#       ax.set_ylabel( y_var )
#       ax.set_zlabel( z_var )
#   
#   if write_mode:
#       outpath = os.path.join(outdir, f"scatter_{space_name}_x-{x_var}_y-{y_var}_hue-{hue_var}_sty-{style_var}.png").replace(" ","")
#       if z_var is not None:
#           outpath = outpath.replace("_hue-", f"_z-{z_var}_hue-")
#       _write_img(fig, outpath)
#       plt.close()
#   else:
#       fig.set_size_inches(fig_size, forward=True)
#       plt.show()
########################################################################################################################
    

# compute secondary statistics
########################################################################################################################
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def pull_data(
        fpath_list = None,
        parent_dir = None, 
        dir_pattern='within_*/permtesting/X_*_dists', 
        f_pattern = '*_vs_*null*',
        enforce_match=True,
        debug=False
        ):
    if fpath_list is None and parent_dir is not None:
        match_pattern = os.path.join(parent_dir, dir_pattern, f"{f_pattern}.csv")
        fpath_list = glob.glob(match_pattern)
        fpath_list.sort()

    if debug:
        print(f"general match pattern is: \n\'{match_pattern}\'")

    if enforce_match:
        print("enforcing modality, feature, and metric matching between data and null")
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries prior to match enforcement.")
        fpath_list = [fpath for fpath in fpath_list if '_'.join(_parse_fpath(fpath, metric=False)) in os.path.basename(fpath)]
        if debug:
            print(f"fpath_list has {len(fpath_list)} entries after match enforcement.")

    alldata_list = [ _load(fpath, enforce_match=enforce_match) for fpath in fpath_list ]

    return alldata_list


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
        default=None,
        help="figure output directory"
    )
    parser.add_argument(
        "-t",
        "--sample_type",
        type=str,
        default="perm",
        help="Specify whether sampling randomness comes from bootstrapping or (indexing) permutation"
    )
    parser.add_argument(
        "-r",
        "--pattern_restriction",
        type=str,
        default=None,
        help="substring pattern to specify subset of matching directories"
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
    
