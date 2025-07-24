import os
import numpy as np
import seaborn as sns
import figutils as futils
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

# global variables
# def_clustermap_vars = ["Wp_XY", "Wp_XYNull_std", "Wp_XYNull_mean"]
def_clustermap_vars = ["Wp_XY", "Wp_XYNull_std"]
def_label_fontsize = 12

def make(alldata_grid, args=None, debug=False):

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

    xnamelist, ynamelist, value_set = futils._get_symmetrized_data(
            alldata_grid, 
            symmetrized_vars=def_clustermap_vars, 
            enforce_symmetry=True,
            set_order=False
            )

    if debug:
        for name in list(value_set.keys()):
            savepath = os.path.join(args.output_dir, f"{name}.csv")
            np.savetxt(savepath, value_set[name])
            print(f"wrote value grid for value \"{name}\" to \"{savepath}\"")
    
    fig_inches = args.fig_size[0] * np.sqrt(70 / len(xnamelist))   # calibrating label fontsize to number of entries
    label_fontsize = def_label_fontsize * np.power(70 / len(xnamelist), 3/4)   # calibrating label fontsize to number of entries

    value_set = futils.get_pval_masks(value_set, alpha = args.alpha)

    # for linkage_var in [None, "Wp_XY"]:
    for linkage_var in [None]:
        generate_clustermaps(
                xnamelist, 
                ynamelist, 
                value_set, 
                linkage_var = linkage_var,
                onelink = True,
                cluster_method = "average",
                alpha = args.alpha,
                log_scale = args.log_scale,
                fig_size = (fig_inches, fig_inches),
                label_fontsize = label_fontsize,
                outdir=args.output_dir,
                write_mode=args.write_mode
                )

    return xnamelist, ynamelist, value_set, alldata_grid


def generate_clustermaps(
        xnamelist,
        ynamelist,
        value_set,
        onelink = True,
        linkage_var = "Wp_XY",
        cluster_method = "average",
        alpha = None,
        log_scale = True,
        fig_size = None,
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
            for i,pval_var in enumerate(pval_vars):
                for show_var in [display_var, pval_var]:
                    if alpha is not None:
                        mask_var = [var for var in mask_vars if pval_var in var][0]     # there should exist a unique entry!
                        mask = value_set[mask_var].copy()
                    else:
                        mask = None
                    if linkage_var is not None:
                        if ("pval" in linkage_var) and ("pval" in display_var):
                            print(f"Skipping \"cluster {display_var} on {linkage_var}\" plot.")
                            continue

                    fig_dict[display_var] = plot_clustermap(
                            xnamelist.copy(),
                            ynamelist.copy(),
                            value_set.copy(),
                            cluster_method = cluster_method,
                            linkage_var = linkage_var,
                            display_var = show_var,
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
        fig_size = None,
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


    # if ("pval" not in display_var) and (alpha is not None):
    if alpha is not None:
        print(f"Masking \'{display_var}\' plot by \'{pval_var}\'...")
        if linkage_var is None:
            outname = f"no-cluster_{display_var}_mask-{pval_var}_alpha{alpha}.png".replace(" ","").replace("0.","")
        else:
            outname = f"cluster-on-{linkage_var}_of-{display_var}_mask-{pval_var}_alpha{alpha}.png".replace(" ","").replace("0.","")
    else:
        if linkage_var is None:
            outname = f"no-cluster_{display_var}.png".replace(" ","")
        else:
            outname = f"cluster-on-{linkage_var}_of-{display_var}.png".replace(" ","")

        
    if np.count_nonzero(np.isnan(display_vals)) > 0:
        print(f"{np.count_nonzero(np.isnan(display_vals))} NaNs removed removed from \'display_vals\' for var \"{display_var}\"")
        np.nan_to_num(display_vals, nan=-1, copy=False)

    if not cluster:
        if mask is None:
            [xnamelist, ynamelist], [display_vals] = futils._reorder_arrays(
                    [xnamelist.copy(), ynamelist.copy()],
                    [display_vals.copy()]
                    )
        else:
            [xnamelist, ynamelist], [display_vals, mask] = futils._reorder_arrays(
                    [xnamelist.copy(), ynamelist.copy()],
                    [display_vals.copy(), mask.copy()]
                    )
        xticklabels = [futils._nice_feats(i.split('_')[0]) for i in xnamelist]
        if all(["Psim" in i for i in ynamelist]):
            yticklabels = [futils._nice_feats(i.split('_')[1]) for i in ynamelist]
        else:
            yticklabels = [futils._nice_feats(' '.join(i.split('_')[1:])) for i in ynamelist]
    else:
        split_char = "\n"   # or, e.g., " "
        xticklabels = [split_char.join(i.split('_',maxsplit=1)) for i in xnamelist]
        yticklabels = [split_char.join(i.split('_',maxsplit=1)) for i in ynamelist]
#   if debug:
#       ### debugging code ###
#       print(f"xticklabels: {xticklabels[0]}")
#       print(f"yticklabels: {yticklabels[0]}")

    if pval_var is not None:
        base_colors = sns.color_palette(n_colors = 3, as_cmap=False)
        #   Lhue=145    # minty light green (in degrees)
        #   Rhue=300    # lavender-lilac (in degrees)
        #   Lrot = -Lhue/360
        #   Rrot = 1 - Rhue/360
        #   light=0.75
        #   dark=0.25
        #   Lcol = hsv_to_rgb(( Lhue/360, 0.5, 0.6 ))   # corresponds to s=60 and l=50 w.r.t. 'diverging_palette' options
        #   Rcol = hsv_to_rgb(( Rhue/360, 0.5, 0.6 ))   # corresponds to s=60 and l=50 w.r.t. 'diverging_palette' options
        if "left-pval" in pval_var:
            # cmap = sns.dark_palette(base_colors[0], reverse=True, as_cmap=True)
            cmap = sns.color_palette("crest", as_cmap=True)
            # cmap = sns.cubehelix_palette(rot=Lrot, light=light, dark=dark, as_cmap=True)
        elif "right-pval"in pval_var:
            # cmap = sns.dark_palette(base_colors[1], reverse=True, as_cmap=True)
            cmap = sns.color_palette("flare", as_cmap=True)
            # cmap = sns.cubehelix_palette(rot=Rrot, light=1-light, dark=1-dark, as_cmap=True)
        else:
            cmap = sns.light_palette(base_colors[2], reverse=False, as_cmap=True)
            # cmap = sns.color_palette("seismic", as_cmap=True)
            # cmap = sns.diverging_palette(Lhue, Rhue, s=60, as_cmap=True)
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



# heatmap plot utility
### is 'mask' enters heatmap as heatmap(mask=mask), then filt=~mask (logical inverse) 
def _disp_logdata(varname, values, diag_val=1e-3, disp_var=True, mask=None):
    if disp_var and "pval" in varname:
        np.fill_diagonal(values, diag_val)
        # np.fill_diagonal(values, np.nan)
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
