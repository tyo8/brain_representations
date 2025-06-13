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
from matplotlib.colors import LinearSegmentedColormap, hsv_to_rgb
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import fdrcorrection

# global variables 

def_fig_size = (24, 24)
def_label_fontsize = 12
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
        xnamelist, ynamelist, value_set, alldata_grid = make_clustermaps(fpath_grid, args=args)

        if args.scatter_plots:
            make_scatterplots(alldata_grid=alldata_grid, args=args)
    elif args.scatter_plots:
        make_scatterplots(fpath_grid=fpath_grid, args=args)

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

    xnamelist, ynamelist, value_set = _get_heatmap_inputs(
            alldata_grid, 
            clustermap_vars=def_clustermap_vars, 
            enforce_symmetry=True
            )

    if debug:
        for name in list(value_set.keys()):
            savepath = os.path.join(args.output_dir, f"{name}.csv")
            np.savetxt(savepath, value_set[name])
            print(f"wrote value grid for value \"{name}\" to \"{savepath}\"")
    
    fig_inches = def_fig_size[0] * np.sqrt(70 / len(xnamelist))   # calibrating label fontsize to number of entries
    label_fontsize = def_label_fontsize * np.power(70 / len(xnamelist), 3/4)   # calibrating label fontsize to number of entries

    value_set = futils.get_pval_masks(value_set, alpha = args.alpha)

    for linkage_var in [None, "Wp_XY"]:
        generate_clustermaps(
                xnamelist, 
                ynamelist, 
                value_set, 
                linkage_var = linkage_var,
                cluster_method = "average",
                alpha = args.alpha,
                log_scale = args.log_scale,
                fig_size = (fig_inches, fig_inches),
                label_fontsize = label_fontsize,
                outdir=args.output_dir,
                write_mode=args.write_mode
                )

    return xnamelist, ynamelist, value_set, alldata_grid

def make_scatterplots(fpath_grid=None, alldata_grid=None, condensed=False, args=None, verbose=True, debug=False):
    if fpath_grid is not None:
        assert args is not None, "if 'fpath_grid' is not None, then 'args' cannot be None either."
        alpha = args.alpha
        alldata_grid = pull_data(fpath_grid, args)
    else:
        assert alldata_grid is not None, "if 'fpath_grid' is None, then 'alldata_grid' must not be None."
        if args is None:
            alpha = None
        else:
            alpha = args.alpha

    allvars = list(alldata_grid[0][0].keys())
    # allvars = [varname for varname in list(alldata_grid[0][0].keys()) if 'two-tail' not in varname]
    _, _, value_set = _get_heatmap_inputs(
            alldata_grid, 
            clustermap_vars=allvars,
            enforce_symmetry=True,
            check_pval=False,
            debug=debug
            )

    if debug:
        print(f"data loaded into 'value_set' has keys: \n{value_set.keys()}")

    value_set = futils.get_pval_masks(value_set, alpha=alpha)
    varlist = list(value_set.keys())
    pval_vars = [var for var in varlist if (('pval' in var) and ('mask' not in var))]
    mask_vars = [var for var in varlist if (('pval' in var) and ('mask' in var))]
    mask_vars = [[var for var in mask_vars if pval_var in var][0] for pval_var in pval_vars]    # forces 'mask_vars' to have same order as 'pval_vars'

    print(varlist)
    dummy_set = {}
    for var in varlist:
        if condensed:
            dummy_set[var] = triu_vals(value_set[var])
        else:
            dummy_set[var] = value_set[var].flatten()

    # if debug:
    #     _debug_value_set(value_set)

    alldata_df = pd.DataFrame(data=dummy_set)
    alldata_df.drop( index = np.where(alldata_df.X_feat_num==0)[0], inplace=True )
    
    if verbose:
        print(f"data shaped into 'alldata_df': \n{alldata_df}")
        print(f"which has keys: \n{alldata_df.columns.values}")
        # alldata_df.to_csv('value_set/alldata_df.csv')
        # np.save('value_set/value_set.npy', value_set, allow_pickle=True)
        print(f"wrote dataframe to: \n{os.path.join(os.getcwd(),'value_set/alldata_df.csv')}")

    fig, outname = do_scatterplot(alldata_df, pval_vars, label_vars=mask_vars, mask_vars=mask_vars, jitter=args.jitter)
    outname.replace('.png', f'alpha{alpha}.png')
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=(12,12))
    # futils._write_img(fig, outpath, fig_size=None)

    wp_vars = [var for var in varlist if "Wp" in var]
    fig, outname = do_scatterplot(alldata_df, wp_vars, zlabel="Wasserstein Distance", label_vars=wp_vars, jitter=args.jitter)
    outname.replace('.png', f'alpha{alpha}.png')
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=(12,12))
    # futils._write_img(fig, outpath, fig_size=None)
    return None

########################################################################################################################

# heatmap plotting
########################################################################################################################
def _get_heatmap_inputs(
        alldata_grid, 
        clustermap_vars=def_clustermap_vars, 
        enforce_symmetry=True, 
        check_pval=True, 
        verbose=True,
        debug=False
        ):
    xnamelist = [i[0]["X_name"].unique()[0] for i in alldata_grid]
    ynamelist = [j["Y_name"].unique()[0] for j in alldata_grid[0]]

    value_set = {}
    if check_pval:
        pval_vars = [ varname for varname in alldata_grid[0][0].columns.values if "pval" in varname ]
        clustermap_vars = clustermap_vars + pval_vars


    for varname in clustermap_vars:
        try:
            vals = np.squeeze(np.array([[j[varname].to_numpy() for j in i] for i in alldata_grid]))
            if "pval" in varname:
                vals = _sym_pvals(vals, varname=varname)
            else:
                vals = _enforce_symmetry(vals, fill_val = 0)
            if verbose:
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
    if verbose:
        print(f"Entries in list of grid values have the following shapes: \n{[value_set[var].shape for var in list(value_set.keys())]}")
        # print("First entry in value_set: ", np.array(value_set[clustermap_vars[0]]))
        print(f"Generating one heatmap for each of the following set of variables: \n{list(value_set.keys())}")
        print("")
        ### debugging code ###

    if enforce_symmetry:
        print(f"enforced symmetry in variables: \'{clustermap_vars}\'.")
        ynamelist = [xnamelist[0], *ynamelist]
        try:
            assert xnamelist == ynamelist
        except AssertionError:
            if debug:
                print(f"namelists are unequal in forced symmetric case! xnamelist: {len(xnamelist)} entries, ynamelist: {len(ynamelist)} entries")
                # print(f"namelists are unequal in forced symmetric case! \nxnamelist: {len(xnamelist)} entries\nynamelist: {len(ynamelist)} entries")
            ynamelist = xnamelist
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
        verbose = True,
        write_mode = True
        ):
    dispvars = list(value_set.keys())
    pval_vars = [var for var in dispvars if (('pval' in var) and ('mask' not in var))]
    mask_vars = [var for var in dispvars if (('pval' in var) and ('mask' in var))]
    dispvars = [var for var in dispvars if 'pval' not in var]

    if onelink and (linkage_var is not None):
        assert linkage_var in dispvars, f"Value does not include variable \"{linkage_var}\", the specified common linkage operator"
        print(f"Using \"{linkage_var}\" as linkage variable while generating clustermaps")
        linkvars = [linkage_var]
    elif linkage_var is None:
        linkvars = [None]
    else:
        print(f"Plotting clustermaps for all (linkage_var, display_var) value pairs (including self-pairs) in {dispvars}")
        linkvars = dispvars

    fig_dict = {}

    if verbose:
        print(f"display vars: {dispvars}")
        print(f"p-value vars: {pval_vars}")

    for linkage_var in linkvars:
        for display_var in dispvars:
            for pval_var in pval_vars:
                mask_var = [var for var in mask_vars if pval_var in var][0]     # there should exist a unique entry!
                mask = value_set[mask_var]
                if linkage_var is not None:
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
                        log_scale = log_scale,
                        fig_size = fig_size,
                        label_fontsize = label_fontsize,
                        outdir = outdir,
                        write_mode = write_mode
                        )

    # can i turn list figure set into something that shows everything?
    return None


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
        log_scale = True,
        label_fontsize = def_label_fontsize,
        fig_size = def_fig_size,
        outdir = None,
        write_mode = True,
        debug = False
        ):

    display_vals = value_set[display_var].copy()

    
    if linkage_var is None:
        xlinkage = None
        cluster = False
    else:
        cluster = True
        import scipy.cluster.hierarchy as hc
        linkage_vals = value_set[linkage_var]
        xlinkage = hc.linkage(squareform(linkage_vals), method=cluster_method, optimal_ordering=True)
        assert linkage_vals.shape==display_vals.shape, "linkage and display values must have same dimensions!"

    if debug:
        print(f"found {np.count_nonzero(xlinkage < 0)} negative linkage values") 
        print(f"found {np.count_nonzero(np.isnan(xlinkage))} NaN linkage values")
        print(f"found {np.count_nonzero(np.isinf(xlinkage))} infinite linkage values")

    print(f"Plotting '{display_var}' values (clustered on \'{linkage_var}\')...")

    cm_title = f"Clustermap plot of {display_var} \n(clustered on {linkage_var})"

    if np.nanmin(mask):
        # i.e., if all mask values are "True"
        print(f"Skipped clustermap. Total masking by pval_var=\'{pval_var}\' at alpha=\'{alpha}\' for: \n\t\t{cm_title}")
        return None

    if log_scale:
        display_var, display_vals, ttl_suffix = _disp_logdata(display_var, display_vals, mask=mask)
        cm_title = cm_title + ttl_suffix


    if ("pval" not in display_var) and (alpha is not None):
        print(f"Masking \'{display_var}\' plot by \'{pval_var}\'...")
        if linkage_var is None:
            outname = f"no-cluster_{display_var}_mask-{pval_var}_alpha{alpha}.png".replace(" ","").replace("0.","")
        else:
            outname = f"cluster-on-{linkage_var}_of-{display_var}_mask-{pval_var}_alpha{alpha}.png".replace(" ","").replace("0.","")
    else:
        outname = f"cluster-on-{linkage_var}_of-{display_var}.png".replace(" ","")

        
    if np.count_nonzero(np.isnan(display_vals)) > 0:
        if debug:
            print(f"{np.count_nonzero(np.isnan(display_vals))} NaNs removed removed from \'display_vals\' for var \"{display_var}\"")
        np.nan_to_num(display_vals, nan=-1, copy=False)

    if not cluster:
        if mask is None:
            [xnamelist, ynamelist], [display_vals] = futils._reorder_arrays(
                    [xnamelist, ynamelist],
                    [display_vals]
                    )
        else:
            [xnamelist, ynamelist], [display_vals, mask] = futils._reorder_arrays(
                    [xnamelist, ynamelist],
                    [display_vals, mask]
                    )
        xticklabels = [futils._nice_feats(i.split('_')[0]) for i in xnamelist]
        yticklabels = [futils._nice_feats(i.split('_')[1]) for i in ynamelist]
    else:
        split_char = "\n"   # or, e.g., " "
        xticklabels = [split_char.join(i.split('_',maxsplit=1)) for i in xnamelist]
        yticklabels = [split_char.join(i.split('_',maxsplit=1)) for i in ynamelist]
#   if debug:
#       ### debugging code ###
#       print(f"xticklabels: {xticklabels[0]}")
#       print(f"yticklabels: {yticklabels[0]}")

    if pval_var is not None:
        Lhue=145    # minty light green (in degrees)
        Rhue=300    # lavender-lilac (in degrees)
        Lrot = -Lhue/360
        Rrot = 1 - Rhue/360
        light=0.75
        dark=0.25
        #   Lcol = hsv_to_rgb(( Lhue/360, 0.5, 0.6 ))   # corresponds to s=60 and l=50 w.r.t. 'diverging_palette' options
        #   Rcol = hsv_to_rgb(( Rhue/360, 0.5, 0.6 ))   # corresponds to s=60 and l=50 w.r.t. 'diverging_palette' options
        if "left-pval" in pval_var:
            # cmap = sns.color_palette("crest", as_cmap=True)
            cmap = sns.cubehelix_palette(rot=Lrot, light=light, dark=dark, reverse=True, as_cmap=True)
        elif "right-pval"in pval_var:
            # cmap = sns.color_palette("magma", as_cmap=True)
            cmap = sns.cubehelix_palette(rot=Rrot, light=1-light, dark=1-dark, as_cmap=True)
        else:
            # cmap = sns.color_palette("seismic", as_cmap=True)
            cmap = sns.diverging_palette(Lhue, Rhue, s=60, as_cmap=True)
    else:
        cmap = sns.color_palette("seismic", as_cmap=True)

    from compare_topostats import _plot_clustermap as _pcl

    g = _pcl(
        display_vals, 
        cluster=cluster,
        cluster_method=cluster_method,
        cm_title = cm_title,
        xticklabels=xticklabels, 
        yticklabels=yticklabels,
        xlinkage=xlinkage,
        ylinkage=xlinkage,
        mask = mask,
        cmap = cmap,
        fig_size=fig_size,
        write_mode=False,
        debug=debug
        )

    fig = g.fig
    ax = g.ax_heatmap
    ax.xaxis.tick_top()
    if not cluster:
        ax.yaxis.tick_left()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)

    if write_mode:
        outpath = os.path.join(outdir, outname)
        futils._write_img(fig, outpath)
        plt.close()
    else:
        fig.set_size_inches(fig_size, forward=True)
        plt.show()

    return g


# actually make the scatterplot
def do_scatterplot(
        df, 
        color_vars, 
        mask_vars=None, 
        label_vars=None, 
        xlabel="X_feat_num", 
        ylabel="Y_feat_num",
        zlabel=None,
        size=75,
        jitter=0.05,
        opacity=0.5,
        log_scale=True,
        verbose=True
        ):
    if zlabel is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    base_colors = sns.color_palette(n_colors = len(color_vars), as_cmap=False)
    # base_colors = sns.color_palette("husl", n_colors = len(color_vars), as_cmap=False)
    color_name = _get_minimal_name(color_vars)

    title = f"{xlabel} vs {ylabel}\ncolored by {color_name}"

    if mask_vars is None:
        mask_vars = [None]*len(color_vars)
        mask_name = None
    else:
        mask_name = _get_minimal_name(mask_vars)
        title += f"\n{mask_name}"
    if label_vars is None:
        label_vars = [None]*len(color_vars)
        label_name = None
    else:
        label_vars = [_clean_scatter_label(var) for var in label_vars]
        label_name = _get_minimal_name(label_vars)
        title += f"\n{label_name}"

    for i, color_var in enumerate(color_vars):
        label_var = label_vars[i]
        mask_var = mask_vars[i]

        if verbose:
            print(f"plotting subset from {mask_var} with coloring from {color_var}")

        if mask_var is None:
            subdf = df
        else:
            subdf = df[ ~df[mask_var] ]

        if 'two-tailed' in color_var:
            cmap = sns.light_palette(base_colors[i], reverse=False, as_cmap=True)
        else:
            cmap = sns.dark_palette(base_colors[i], reverse=True, as_cmap=True)

        color = subdf[color_var].to_numpy().astype(float)
        x = subdf[xlabel].to_numpy().astype(float)
        y = subdf[ylabel].to_numpy().astype(float)

        if log_scale:
            if 'pval' in color_var:
                color = np.log(color)
            x = np.log10(x)
            y = np.log10(y)

        min_cval = np.nanmin(color)
        max_cval = np.nanmax(color)
        np.nan_to_num(color, copy=False, neginf=min_cval, posinf=max_cval)

        x *= 1+(1/2 - np.random.rand(*x.shape))*jitter*2
        y *= 1+(1/2 - np.random.rand(*y.shape))*jitter*2

        if zlabel is None:
            ax.scatter(
                    x=x,
                    #x=np.log(y/x),
                    y=y,
                    #y=color,
                    s=size,
                    c=color, 
                    cmap=cmap,
                    alpha=opacity,
                    linewidths=0,
                    label=label_var
                    )
        else:
            if 'std' in color_var:
                zmean = subdf[color_vars[i-1]].to_numpy()
                ax.errorbar(
                        x,      # xs
                        y,      # ys
                        zmean,  # zs
                        zerr=color,
                        color=base_colors[i], 
                        alpha=opacity,
                        fmt='none'      # plots errorbars only, no datapoints
                        )
            else:
                ax.scatter(
                        x,      # xs
                        y,      # ys
                        color,  # zs
                        s=size/3,
                        color=base_colors[i], 
                        alpha=opacity,
                        linewidths=0,
                        label=label_var
                        )

    leg = ax.legend(loc="best")
    # face_colors = ['blue', 'orange', 'green']
    for i,handle in enumerate(leg.legend_handles):
        print(f"{i}-th pre-update legend marker face color:", handle.get_facecolor())
        handle.set_facecolor(base_colors[i])
        handle.set_alpha(1)
        print(f"{i}-th post-update legend marker face color:", handle.get_facecolor())
        # handle.set_facecolor(face_colors[i])
    ax.set_xlabel(_clean_scatter_label(xlabel))
    ax.set_ylabel(_clean_scatter_label(ylabel))
    if zlabel is not None:
        ax.set_zlabel(_clean_scatter_label(zlabel))


    outname = f"x-{xlabel}_y-{ylabel}_c-{color_name}.png"
    if log_scale:
        outname.replace('.png', '_log.png')
    if mask_name is not None:
        outname.replace('.png', '_m-{mask_name}.png')
    if label_name is not None:
        outname.replace('.png', '_label-{label_name}.png')
    return fig, outname

# scatterplot title utility
def _get_minimal_name(namelist):
    nameset = [set(i) for i in namelist]
    minlist = list(nameset[0].intersection(*nameset))
    minlist.sort()
    min_name = ''.join(minlist)
    return min_name

# scatterplot labelling utility
def _clean_scatter_label(label):
    clean_label = label
    if 'left-pval' in label:
        clean_label = "Convergent"
    elif 'right-pval' in label:
        clean_label = "Divergent"
    elif 'two-tailed' in label:
        clean_label = "Incomparable"

    if 'fdr' in label:
        clean_label += " (FDR)"
    elif 'fwe' in label:
        clean_label += " (FWE)"

    if 'subject' in label:
        clean_label = clean_label.replace("(F", "(subject-null F")
    elif 'feature' in label:
        clean_label = clean_label.replace("(F", "(feature-null F")

    if 'Wp_XY' in label:
        clean_label = label.replace('_XY', '(X,Y) ').replace('_',' ')

    if '_feat_num' in label:
        clean_label = label.replace('_feat_num',' feature number')
    return clean_label 


# heatmap plot utility
### is 'mask' enters heatmap as heatmap(mask=mask), then filt=~mask (logical inverse) 
def _disp_logdata(varname, values, disp_var=True, mask=None):
    if disp_var and "pval" in varname:
        np.fill_diagonal(values, np.nan)
        title_suffix = " (-log10(2p))"
        values = -np.log10(2*values)
        nanval = -1

    if "Wp_XY" in varname:
        values[ values==0 ] = np.nan
        title_suffix = " (log10(W_p))"
        values= np.log10(values)
        if mask is None:
            filt = np.ones(values.shape, dtype='bool')
        elif np.nanmin(mask):   
            # i.e., if mask has only True values
            filt = np.ones(values.shape, dtype='bool')
        else:
            filt = ~mask
        nanval = -1.1*np.nanmax(np.abs(values[filt]))

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
#   args.dir_pattern='X_*_dists'
#   args.f_pattern = '*_vs_*.csv'
#   if args.pattern_restriction is not None and args.permtype is not None:
#       if not args.output_dir.endswith(args.pattern_restriction):
#           args.output_dir = os.path.join(args.output_dir, args.pattern_restriction)
#
#       args.dir_pattern = args.dir_pattern.replace('_dists', f'*{args.pattern_restriction}*_dists')
#       args.f_pattern = args.f_pattern.replace('_vs_', f'{args.pattern_restriction}*_vs_*{args.pattern_restriction}')
#   if args.permtype is not None:
#       args.f_pattern = args.f_pattern.replace(".csv",f"{args.permtype}Perms.csv")
#
#   pdir_pattern = os.path.join( args.input_dir, args.dir_pattern )
#
#   dpath_list = glob.glob(pdir_pattern); dpath_list.sort()
#   fpath_grid = [ glob.glob(os.path.join(dpath, args.f_pattern)) for dpath in dpath_list ]
#   fpath_grid = [ pathlist for pathlist in fpath_grid if pathlist ]    # removes empty lists (corresponding to directories with no successful search hits)
#   [pathlist.sort() for pathlist in fpath_grid]

    fpath_grid = futils._get_fpath_set(args, dist_type="pair", set_type="grid")
    fpath_list = list(itertools.chain(*fpath_grid))

    if args.verbose:
        print(f"Matching patterns of general form: \n{(args.pdir_pattern, args.f_pattern)}:")
        print(f"\tshaping matches into a \'filepath grid\' array results in shape(s): { ( len(fpath_grid), list(set( [ len(i) for i in fpath_grid ] )) ) }")
        print(f"\tfound {len(fpath_list)} total matches.")

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
    if mtx.shape[0] == mtx.shape[1]:
        sym_mtx = (mtx + mtx.T)/2
    else:
        assert (mtx.shape[0]-1)==mtx.shape[1], f"Input matrix assumed to have shape (n,n-1): instead, given matrix has shape {mtx.shape}"

        # takes values from upper diagonal
        sym_mtx = squareform(triu_vals(mtx, k=0))
        np.fill_diagonal(sym_mtx, fill_val)

        if isinstance(sym_mtx[0][1], float):
            assert np.allclose(sym_mtx, sym_mtx.T, equal_nan=True), f"Symmetrization failed: \"sym_mtx\" is \n{sym_mtx}"

    return sym_mtx

def _sym_pvals(pval_mtx, varname=None):
    if varname is None:
        fill_val = 0
    else:
        fill_val = int("right" in varname)

        sym_pval = _enforce_symmetry(pval_mtx, fill_val=0)
        if "fdr" in varname:
            sym_pval = squareform(fstats.correct_pvals(squareform(sym_pval), corr_type="fdr"))
        elif "fwe" in varname:
            sym_pval = squareform(fstats.correct_pvals(squareform(sym_pval), corr_type="fwe"))
        np.fill_diagonal( sym_pval, fill_val )
        return sym_pval

def triu_vals(A, k=1):
    n = min(A.shape)
    vals = A[np.triu_indices(n, k)]
    return vals
########################################################################################################################


## Debugging functions
########################################################################################################################
def _debug_value_set(value_set):
    # del value_set['datatype']
    varlist = list(value_set.keys())
    print(f"data loaded into 'value_set' has (upper-triangular) shapes: \n{[(var, value_set[var].shape) for var in varlist]}")
    # print(f"data loaded into 'value_set' has first values: \n{[(var, value_set[var][0]) for var in varlist]}")
    outdir = "value_set"
    with open('value_set/value_set.npy','wb') as fout:
        np.save(fout, value_set)
    for var in varlist:
        fpath = os.path.join(outdir, f"{var}.txt")
        val = triu_vals(value_set[var])
        np.savetxt(fpath, val)
        print(f'{var} written to file:', os.path.join(os.getcwd(), fpath))
        value_set[var] = val
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

