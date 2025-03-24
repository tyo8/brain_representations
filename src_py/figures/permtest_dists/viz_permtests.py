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

# global variables 

def_fig_size = (24, 24)

def_pattern='*X_*_dists'


# def_heatmap_vars = ["Wp_XY", "empirical_pval"]
def_heatmap_vars = ["empirical_pval"]
def_scatter_vars = ["Wp_XY", "Y_type"] 

# exp_outtype="All_vs_AllNull/X_ICA15_Amps_Psim_dists/ICA15_Amps_Psim_vs_Schaefer100_Amps_Psim_null-subjectPerms.csv"
modalities = ["Glasser", "ICA", "grad", "Schaefer", "PROFUMO", "Yeo"]

############################################ FIGURE MAKING FUNCTIONS ###################################################
########################################################################################################################
# make quick and dirty paired-null distance distribution summaries
########################################################################################################################
def one_pair_plot(fpath, fig_title=None, verbose=True, debug=False):
    full_df = pd.read_csv(fpath, index_col=0)
    data_df = full_df[ full_df["datatype"]~="Null" ]
    null_df = full_df[ full_df["datatype"]=="Null" ]

    # compute p-value from 2-sided test against empirical CDF (enforcing inf(p)=1/N)
    data_pval = 1 - np.mean(data_df > null_df["Wp_XY"].to_numpy())
    data_pval = 2*min(data_pval, 1 - data_pval)
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
def _get_heatmap_inputs(alldata_grid, heatmap_vars=def_heatmap_vars, debug=True):
    xnamelist = [list(set(i[0]["X_type"]))[0] for i in alldata_grid]
    ynamelist = [list(set(j["Y_type"]))[0] for j in alldata_grid[0]]

    valuegrid_list = [None]*len(heatmap_vars)

    for idx, varname in enumerate(heatmap_vars):
        try:
            valuegrid_list[idx] = np.squeeze(np.array([[j[varname].dropna().to_numpy() for j in i] for i in alldata_grid]))
        except ValueError:
            new_entry = [[j[varname].dropna().to_numpy() for j in i] for i in alldata_grid]
            valuegrid_list[idx] = np.squeeze(new_entry)
            if debug:
                ### debugging code ###
                print(f"found data inmogeneity in {varname} readin. attempted new entry has data of following shapes and values:")
                print([var.shape for var in new_entry])
                print("corresponding to pairs:")
                print([[(j["X_type"],j["Y_type"]) for j in i] for i in alldata_grid])
                # print(new_entry)
                ### debugging code ###
            

    if debug:
        ### debugging code ###
        print(f"Names of 'X' spaces: \n{xnamelist}")
        print(f"Names of 'Y' spaces: \n{ynamelist}")
        print(f"Entries in list of grid values have the following shapes: \n{[i.shape for i in valuegrid_list]}")
        print("First entry in valuegrid_list: ", np.array(valuegrid_list[0]))
        print(f"Generating one heatmap for each of the following set of variables: \n{heatmap_vars}")
        print("")
        ### debugging code ###
    return xnamelist, ynamelist, valuegrid_list


def generate_heatmap_plots(
        xnamelist, 
        ynamelist, 
        valuegrid_list, 
        heatmap_vars = def_heatmap_vars,
        outdir = None,
        name_type = "exp_results",
        self_cluster = False,
        cluster_method = "average",
        write_mode = True
        ):
    for i, value_grid in enumerate(valuegrid_list):
        value_name = heatmap_vars[i]
        if value_grid.ndim > 2:
            mean_grid = np.mean(value_grid, axis=2)
            heatmap_plot(
                    f"{value_name}_mean", 
                    mean_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )
            std_grid = np.std(value_grid, axis=2)
            heatmap_plot(
                    f"{value_name}_stddev", 
                    std_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )
        else:
            heatmap_plot(
                    value_name, 
                    value_grid, 
                    xnamelist, 
                    ynamelist,
                    outdir=outdir,
                    name_type=name_type,
                    self_cluster=self_cluster,
                    write_mode=write_mode
                    )



def heatmap_plot(
        value_name, 
        value_grid, 
        xnamelist, 
        ynamelist, 
        outdir=None,
        name_type="exp_results", 
        self_cluster=True,
        cluster_method="average",
        label_fontsize = 8,
        write_mode=True,
        debug=False
        ):

    print(f"Plotting grid of '{value_name}' values...")

    xticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in xnamelist]
    yticklabels = ["\n".join(i.split('_',maxsplit=1)) for i in ynamelist]
    
    vartype = _get_vartype(xticklabels[0].split('\n'), name_type=name_type)
    rb_title = f"Heatmap plot of {value_name}"

    if "pval" in value_name:
        value_grid[np.abs(value_grid) == np.inf] = np.nan
        value_grid = -np.log10(2*value_grid)
        rb_title = rb_title + " (-log10(2p))"
        

    if debug:
        ### debugging code ###
        print(f"xticklabels: \n{xticklabels}")
        print(f"yticklabels: \n{yticklabels}")

        value_uniq = np.unique((value_grid + value_grid.T)/2)
        print(f"Variable 'value_grid' has shape: {value_grid.shape}")
        print(f"Variable 'value_grid' is roughly symmetric: {np.allclose(value_grid, value_grid.T)}.")
        print(f"Variable 'value_grid' is strictly symmetric: {np.all(np.equal(value_grid, value_grid.T))}.")
        print(f"Variable 'value_grid' contains {len(value_uniq)} unique elements:")
        print(np.histogram(value_uniq))
        print("")
        ### debugging code ###

#   if np.allclose(value_grid, value_grid.T):
#       value_grid = (value_grid + value_grid.T)/2      # force exact symmetry because clustermap symmetry tolerance is stricter than np.allclose
#   else:
#       raise ValueError("Variable 'value_grid' must be (at least approximately) symmetric to use as linkage for a clustermap")

    fig, ax = plt.subplots()
    ax = sns.heatmap(
            value_grid, 
            square = True, 
            cbar = True, 
            ax=ax, 
            cmap = sns.color_palette("Spectral", as_cmap=True),
            xticklabels=xticklabels, 
            yticklabels=yticklabels
            )
    ax.set(title = rb_title)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    if write_mode:
        outpath = os.path.join(outdir, f"heatmap_{value_name}.png").replace(" ","")
        _write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()
########################################################################################################################


# heatmap plot utilities
########################################################################################################################
def _construct_title(nametuple, title_type=None):
    if title_type=="heatmap":
        title = "Variation of %s \n in %s and %s \n over %s pairs" % nametuple
    elif title_type=="scatter":
        if len(nametuple) == 3:
            title = "Relationships between resampling stability measurements in %s:\n%s vs. %s" % nametuple
        elif len(nametuple) == 4:
            title = "Relationships between resampling stability measurements in %s:\n%s vs. %s vs. %s" % nametuple
        else:
            raise IOError(f"Input 'nametuple'={nametuple} must have len=3 or len=4 for title of type {title_type}")
    else:
        raise IOError(f"Unrecognized title type '{title_type}'")
    return title

def _get_vartype(ticklabel, name_type="exp_results", debug=True):

    if debug:
        ### debugging code ###
        print(f"Submitting ticklabels that look like: \n{ticklabel}")
        ### debugging code ###

    if isinstance(ticklabel, list):
        vartype = [None]*len(ticklabel)
        for i,name in enumerate(ticklabel):
            vartype[i] = _name_vartype(name, name_type=name_type)
        vartype = tuple(vartype)
    else:
        vartype = _name_vartype(ticklabel, name_type=name_type)
    return vartype

def _name_vartype(name, name_type="exp_results", debug=True):
    if debug:
        ### debugging code ###
        print(f"Producing variable type from name '{name}' under '{name_type}' conventions.")
        ### debugging code ###
    if name_type=="exp_results":
        if "Psim" in name or name=="inner" or name=="geodesic":
            vartype = "Dissimilarity Function"
        elif name=="Maps" or name=="Amps" or name=="pNMs" or name=="NMs":
            vartype = "Feature Type"
        elif any([mode in name for mode in modalities]):
            vartype = "Modality"
        else:
            raise IOError(f"Variable \"{name}\" not of type recognized under \"{name_type}\" conventions")
    else:
        raise IOError(f"Unrecognized name_type \"{name_type}\"")

    if debug:
        ### debugging code ###
        print(f"variable type name: {vartype}")
        ### debugging code ###
    return vartype

def _manual_stylemap(uniq_data, style_order=None, debug=True):
    if style_order is None:
        style_order = list(matplotlib.markers.MarkerStyle("").markers.keys())
    mdict = {}
    for i,val in enumerate(uniq_data):
        mdict[val] = style_order[i]

    if debug:
        ### debugging code ###
        print("Style order:", style_order)
        print("Marker dictionary:", mdict)
        ### debugging code ###
    return mdict

def _manual_colormap(uniq_data, debug=True):
    cvec = np.linspace(0, 1, len(uniq_data))
    cdict = {}
    for i, val in enumerate(uniq_data):
        cdict[val] = cvec[i]
    
    if debug:
        ### debugging code ###
        print("Color dictionary:", cdict)
        ### debugging code ###
    return cdict
########################################################################################################################
########################################################################################################################


# compute secondary statistics
########################################################################################################################
# add an empirical p-val to a dataframe containing only a single data-derived distance and its null counterparts
def _add_emp_pval(df, check_match=True):
    datarow = df[df["datatype"] == "Data"]
    nullrows = df[df["datatype"] == "Null"]

    Wp_XY = datarow["Wp_XY"].to_numpy()
    Wp_XYnull = nullrows["Wp_XY"].to_numpy()

    if len(Wp_XYnull) == 0:
        empirical_pval = -np.inf
    else:
        if check_match:
            check_cols = [col for col in df.columns if col.startswith("X") or col.startswith("Y")]
            err_str =  f"data row and null rows are not of matching type: \n{[[col, set(datarow[col]), set(nullrows[col])] for col in check_cols]}"
            assert all( [ set(datarow[col]) == set(nullrows[col]) for col in check_cols ] ), err_str

        prop_lower = np.mean(Wp_XY > Wp_XYnull)
        # compute p-value from 2-sided test against empirical CDF (enforcing inf(p)=1/N)
        empirical_pval = max(1/len(Wp_XYnull), 2*min(prop_lower, 1 - prop_lower))
        # compute p-value from 1-sided test against empirical CDF (enforcing inf(p)=1/N)
        # empirical_pval = max(1/len(Wp_XYnull), 1 - prop_lower)

    df["empirical_pval"] = [empirical_pval] + [np.nan]*len(Wp_XYnull)

    return df
########################################################################################################################


# Data wrangling functions
########################################################################################################################
def pull_data(parent_dir, dir_pattern='X_*_dists', f_pattern = '*_vs_*', name_type="exp_results", enforce_sym=True, debug=True):
    dirlist = glob.glob(os.path.join(parent_dir, dir_pattern))
    dirlist.sort()

    fpath_grid = [ glob.glob(os.path.join(X_dir, f"{f_pattern}.csv")) for X_dir in dirlist ]
    [i.sort() for i in fpath_grid]

    alldata_grid = [ [ _load(fpath, name_type=name_type) for fpath in X_sublist ] for X_sublist in fpath_grid ]
    gridlist_shape = [len(alldata_grid), set([len(i) for i in alldata_grid]), set([i.shape for j in alldata_grid for i in j])]

    if debug:
        ### debugging code ###
        print(f"Pulling from fpath_grid w/ 00 entry: \n{fpath_grid[0][0]}")
        if not isinstance(alldata_grid[0], list):
            print(f"alldata_grid loadin variable is not nested lists, but instead has following structure: \n{[type(x) for x in alldata_grid]}")
        try:
            print(f"00 entry of alldata_grid: \n{alldata_grid[0][0]}")
        except IndexError:
            print(f"0-row entry of alldata_grid: \n{alldata_grid[0]}")
        ### debugging code ###

    if enforce_sym:
        # Enforces symmetry under assumption 'alldata_grid' produced by a pairwise process skipping its first trivial pairing
        alldata_grid = _enforce_symmetry(alldata_grid, debug=debug)

    return alldata_grid


def _load(input_fpath, name_type="exp_results", check_pval=True, parse_longname=False):
    if name_type=="exp_results":

        data_df = pd.read_csv(input_fpath, index_col=0)

        if parse_longname:
            data_df[["X_mod","X_feat","X_diff"]] = data_df["X_type"].str.split('_', n=2, expand=True)
            data_df[["Y_mod","Y_feat","Y_diff"]] = data_df["Y_type"].str.split('_', n=2, expand=True)
            data_df.drop(["X_type","Y_type"], axis=1, inplace=True)


        if check_pval:
            if "empirical_pval" not in data_df.columns:
                data_df = _add_emp_pval(data_df)

    return data_df


# Enforces symmetry under assumption 'gridlist' produced by a pairwise process skipping its first trivial pairing
def _enforce_symmetry(gridlist, debug=True):

    print("Enforcing symmetry under assumption \'gridlist\' produced by pairwise process that skipped its first trivial pairing...")

    for i in range(1,len(gridlist)):
        gridlist[i].insert(0, gridlist[0][i-1])
    
    # NOTE: assumes each data-layer entry in gridlist has dataframe format!
    df = gridlist[0][0].copy()

    if debug:
        print(f"first trivial pairing before substitions: \n{df}")
    
    for var in (def_heatmap_vars + def_scatter_vars):
        val = df[var].dropna().to_numpy()
        df[var].replace(to_replace=val, value=-np.inf, inplace=True)
    
    check_cols = [col for col in df.columns if col.startswith("X")]
    for col in check_cols:
        df[col.replace("X","Y")] = df[col].copy()

    if debug:
        print(f"first trivial pairing after substitions: \n{df}")

    gridlist[0].insert(0, df)

    gridlist_shape = [len(gridlist), set([len(i) for i in gridlist]), set([i.shape for j in gridlist for i in j])]
    if debug:
        print(f"After enforcing symmetry under pairwise process assumption, \'gridlist\' has \"shape\" {tuple(gridlist_shape)}.")

    return gridlist


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
        default="",
        help="directory with name of type []_vs_[] containing bootstrapped distance outputs"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="",
        help="figure output directory"
    )
    parser.add_argument(
        "-n",
        "--name_type",
        type=str,
        default="exp_results",
        help="Specifies the class of naming conventions used for the target data"
    )
    parser.add_argument(
        "-p",
        "--pattern",
        type=str,
        default=def_pattern,
        help="substring pattern to specify subset of matching directories"
    )
    parser.add_argument(
        "-R",
        "--do_heatmap",
        default=False,
        action="store_true",
        help="Generate heatmaps of pairwise summary comparisons over varying parameters in each pair"
    )
    parser.add_argument(
        "-S",
        "--do_scatter",
        default=False,
        action="store_true",
        help="Generate scatterplots of per-space stability summary quantities"
    )
    parser.add_argument(
        "-w",
        "--write_mode",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    args = parser.parse_args()

    alldata_grid = pull_data(args.input_dir, dir_pattern=args.pattern, name_type=args.name_type)

    if "_vs_self" in args.input_dir:
        self_cluster=True
    else:
        self_cluster=False

    if not os.path.isdir(args.output_dir):
        print(f"Warning: making new directory {output_dir}")
        os.mkdir(args.output_dir)

    if args.do_heatmap:
        xnamelist, ynamelist, valuegrid_list = _get_heatmap_inputs(alldata_grid, heatmap_vars=def_heatmap_vars)
        generate_heatmap_plots(
                xnamelist, 
                ynamelist, 
                valuegrid_list, 
                heatmap_vars=def_heatmap_vars,
                outdir=args.output_dir,
                name_type=args.name_type,
                self_cluster=self_cluster,
                cluster_method=None,
                write_mode=args.write_mode
                )

    if args.do_scatter:
        scatter_df, hue_var, style_var = _get_scatter_df(alldata_grid, scatter_vars=def_scatter_vars, name_type=name_type)
        generate_scatter_plots(
                args.output_dir, 
                scatter_df, 
                hue_var=hue_var, 
                style_var=style_var, 
                write_mode=args.write_mode
                )

