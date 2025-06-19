import os
import numpy as np
import pandas as pd
import seaborn as sns
import figutils as futils
from matplotlib import pyplot as plt

markerlist = ["<", ">", "^", "v", "P", "D", "s", "P", "d", "."]

def make(alldata_grid, args=None, verbose=True, debug=False):
    assert alldata_grid is not None, "if 'fpath_grid' is None, then 'alldata_grid' must not be None."
    if args is None:
        alpha = None
    else:
        alpha = args.alpha

    allvars = list(alldata_grid[0][0].keys())
    # allvars = [varname for varname in list(alldata_grid[0][0].keys()) if 'two-tail' not in varname]
    xnamelist, ynamelist, value_set = futils._get_symmetrized_data(
            alldata_grid, 
            symmetrized_vars=allvars,
            enforce_symmetry=True,
            check_pval=False,
            debug=debug
            )

    if debug:
        print(f"data loaded into 'value_set' has keys: \n{value_set.keys()}")

    value_set = futils.get_pval_masks(value_set, alpha=alpha)

    varlist = list(value_set.keys())
    pval_vars = [var for var in varlist if (('pval' in var) and ('mask' not in var))]
    if alpha is None:
        mask_vars = None
    else:
        mask_vars = [var for var in varlist if (('pval' in var) and ('mask' in var))]
        mask_vars = [[var for var in mask_vars if pval_var in var][0] for pval_var in pval_vars]    # forces 'mask_vars' to have same order as 'pval_vars'

    if debug:
        print(varlist)

    alldata_df = pd.DataFrame(data={k: v.flatten() for k,v in value_set.items()})
    alldata_df.dropna(axis='index', inplace=True) 
    
    if debug:
        print(f"data shaped into 'alldata_df': \n{alldata_df}")
        print(f"which has keys: \n{alldata_df.columns.values}")
        alldata_df.to_csv('value_set/alldata_df.csv')
        np.save('value_set/value_set.npy', value_set, allow_pickle=True)
        print(f"wrote dataframe to: \n{os.path.join(os.getcwd(),'value_set/alldata_df.csv')}")

    fig, outname = do_scatterplot(alldata_df, pval_vars, label_vars=mask_vars, mask_vars=mask_vars, jitter=args.jitter)
    outname = outname.replace('.png', f'_alpha{alpha}.png').replace('0.','')
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=args.fig_size)
    # futils._write_img(fig, outpath, fig_size=None)

    wp_vars = [var for var in varlist if "Wp" in var]
    fig, outname = do_scatterplot(alldata_df, wp_vars, zlabel="Wasserstein Distance", label_vars=wp_vars, jitter=args.jitter)
    outname = outname.replace('.png', f'_alpha{alpha}.png').replace('0.','')
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=args.fig_size)
    # futils._write_img(fig, outpath, fig_size=None)
    return xnamelist, ynamelist, value_set


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
            print(f"scatter-plotting subset from {mask_var} with coloring from {color_var}")

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
            if zlabel is not None:
                color = np.log10(color)
            x = np.log10(x)
            y = np.log10(y)

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
                    marker = markerlist[i],
                    cmap=cmap,
                    alpha=opacity,
                    linewidths=0,
                    label=label_var
                    )
        else:
            if 'std' in color_var:
                zmean = subdf[color_vars[i-1]].to_numpy()
                zerr = color
                if log_scale:
                    zmean, zerr = _get_log_errorbars(zmean, log_err=color)
                ax.errorbar(
                        x,      # xs
                        y,      # ys
                        zmean,  # zs
                        zerr=zerr,
                        ecolor=base_colors[i-1], 
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
                        marker = markerlist[i],
                        alpha=opacity,
                        linewidths=0,
                        label=label_var
                        )

    leg = ax.legend(loc="best")
    # face_colors = ['blue', 'orange', 'green']
    for i,handle in enumerate(leg.legend_handles):
        # print(f"{i}-th pre-update legend marker face color:", handle.get_facecolor())
        handle.set_facecolor(base_colors[i])
        handle.set_alpha(1)
        # print(f"{i}-th post-update legend marker face color:", handle.get_facecolor())
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

def _get_log_errorbars(meanvals, log_err=-np.inf, base=10):
    err = np.power(base, log_err)
    vals_range = np.array( [meanvals - err, meanvals + err] )
    log_range = np.log(vals_range) / np.log(base)
    log_mean = np.log(meanvals) / np.log(base)
    log_err = np.abs(log_range - log_mean)
    return log_mean, log_err


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
