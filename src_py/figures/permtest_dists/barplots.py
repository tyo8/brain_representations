import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import figutils as futils
from matplotlib import patches as pch
from matplotlib import pyplot as plt

################
# adapted from the "Circular barplot with gorups in Matplotlib" post
# from the Python graph gallery:
# 
#       <https://python-graph-gallery.com/circular-barplot-with-groups/>
################

def_fig_size=(12,12)
def_label_fontsize=8
name_limit = 5
unq_pair_limit = 200
# unq_pair_limit = 332

def make( results_list, args=None ):
    if args is None:
        output_dir = '.'
        fig_size = def_fig_size
    else:
        output_dir = args.output_dir
        fig_size = args.fig_size

    for chi2_result in results_list:
        varname = chi2_result["obs_type"]
        df = chi2_result["observed"]

        if len(df) > unq_pair_limit:
            print(f"Too many {varname} results to visualize: {len(df)}. Skipping.")
            continue
        elif len(df) == 1:
            print(f"Trivial case: {varname} has only 1 entry. Skipping visualization.")
            continue
        elif 2*len(df) > name_limit**2 - name_limit:
            axtitle_fontsize = def_label_fontsize * name_limit / np.sqrt(2*len(df))
        else:
            axtitle_fontsize = def_label_fontsize * 1.5

        stats_tag = df.columns.values[0].replace("Convergent_",'').replace('-null','')

        suptitle, axtitles = _get_titles( chi2_result )

        if not len(axtitles) == len(df):
            Warning("axis titles have a different length ({len(axtitles)}) than dataframe ({len(df)}) for \'{varname}\'.")
            print(f"\taxis titles: \n{axtitles}")
            print(f"\tdataframe: \n{df}")

        fig, axgrid = sym_subradplots(
                df.drop( columns=[varname] ), 
                titles=axtitles,
                title_fontsize=axtitle_fontsize,
                fig_size=fig_size
                )
        fig.suptitle(suptitle)

        outname = f"counts-barplot-stats_{varname}_{stats_tag}.png"
        outpath = os.path.join(output_dir, outname)
        futils._write_img(fig, outpath, fig_size=fig_size)

    return None

def sym_subradplots( 
                    df, count_vars=None, 
                    titles=None, title_fontsize=def_label_fontsize,
                    fig_size=def_fig_size, 
                    debug=False
                    ):

    df.index = df.index.map(ast.literal_eval)
    pair_labels = df.index.to_list()
    names = sorted(np.unique( list(zip(*pair_labels))[0] ).tolist())

    # toggle some display options to prevent overcrowding in many-category case
    if len(names) <= name_limit:
        tangential_labels = True
        make_legend = True
        # make_legend = False
        rotation = None
        both_axes = True
    else:
        tangential_labels = False
        make_legend = True
        rotation = 90
        both_axes = False

    if debug:
        print(f"pair labels: \n{pair_labels}")
        print(f"unique nameset: \n{names}")

    if count_vars is None:
        count_vars = [ var for var in df.columns.values if any(
            [ x in var for x in ["Convergent", "Divergent", "Incomparable"] ]
            ) ]

    fig, axgrid = plt.subplots(
            nrows = len(names),
            ncols = len(names),
            figsize=fig_size, 
            subplot_kw={"projection": "polar"}
            )

    for i in range(len(names)):
        for j in range(len(names)):
            ax = axgrid[i,j]
            pair_key = tuple(sorted((names[i], names[j])))
            if pair_key not in df.index:
                ax.set_visible(False)
                continue

            if titles is None:
                title=None
            elif isinstance(titles, str):
                title=titles
            elif isinstance(titles, list):
                pair_idx = pair_labels.index(pair_key)
                title = titles[pair_idx]

            series = df[count_vars].T[pair_key]
            radplot_fromseries( 
                               series, 
                               label_name="label",
                               tangential_labels=tangential_labels,
                               ax=ax
                               )
            if i==len(names)-1:
                ax.set_xlabel(names[j], fontsize=def_label_fontsize*2, rotation=rotation)
            if j==0 and both_axes:
                ax.set_ylabel(names[i], fontsize=def_label_fontsize*2)

            ax.set_title(title, fontsize=title_fontsize)

    if make_legend:
        handles, labels = futils.proxy_legend(
                _get_label_set(count_vars), 
                palette=None
                )
        fig.legend(handles, labels, loc='upper left')

    return fig, axgrid


def radplot_fromseries( 
                       series, label_name="label", 
                       tangential_labels=True,
                       title="default", ax=None
                       ):
    label_df = _get_label_df(series, label_name=label_name)

    ax = radial_barplot(
            label_df,
            name=label_name, 
            count=series.name, 
            group=label_name,
            pad_ct=0,
            tangential_labels=tangential_labels,
            ax=ax
            )
    if title=="default":
        ax.set_title(f"Comparison: {series.name}")
    else:
        ax.set_title(title)
    return None
    

def radial_barplot(df, 
                   name="name", count="value", group="group",
                   palette=None, fig_size=def_fig_size,
                   phase=0, pad_ct=0, ax=None,
                   tangential_labels=False,
                   seaborn_theme=True
                   ):

    if seaborn_theme:
        sns.set_theme()

    labels = df[name]
    values = df[count]
    groups = df[group].unique()

    # size of each group
    group_cts = [len(i[1]) for i in df.groupby(group)]

    colorset = sns.color_palette(
            palette=palette, 
            n_colors = len(groups),
            as_cmap=False
            )
    angles = np.linspace(0, 2*np.pi, num=len(values)+pad_ct*len(groups), endpoint=False)

    idx_offset = 0
    idx = []
    for sz in group_cts:
        idx += list(range(idx_offset + pad_ct, idx_offset + sz + pad_ct))
        idx_offset += sz + pad_ct

    if ax is None:
        fig,ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": "polar"})

    ax.set_theta_offset(phase)
    ax.set_ylim(-max(values)/3, max(values)*1.05)
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # using different colors for each group
    colors = [colorset[i] for i, sz in enumerate(group_cts) for _ in range(sz)]

    # add bars to plot
    ax.bar(
            angles[idx], 
            values, 
            width=(2*np.pi/len(angles)),
            color=colors,
            edgecolor="white",
            linewidth=2
            )

    if name == group:
        if tangential_labels:
            # labels = [f"{i[1]} {i[0]}" for i in zip(labels, list(map(str,values)))]
            labels = list(map(str,values))
        else:
            # labels = list(map(str,values))
            labels = [None]*len(values)

    _add_labels(
            angles=angles[idx], 
            values=values, 
            labels=labels, 
            phase=phase, 
            ax=ax,
            tangential_labels=tangential_labels
            )

    if name != group:
        labels=values
        _add_group_text(
                angles=angles,
                groups=groups,
                group_cts=group_cts,
                pad_ct=pad_ct,
                ax=ax
                )
    return ax

############################################ HELPER FUNCTIONS ###################################################
########################################################################################################################
def _get_label_rotation(angle, phase, tangential_labels=True):
    # rotation must be specified in degrees
    rotation = np.rad2deg(angle + phase)
    if tangential_labels:
        rotation = rotation + 90
        alignment = "center"
        if angle <= np.pi:
            rotation = rotation + 180
    elif angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else:
        alignment = "left"

    return rotation, alignment

def _add_labels(angles, values, labels, phase, ax, tangential_labels=True):
    # This is the space between the end of the bar and the label
    padding = 4
    delta_ang = np.diff(angles)
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        
        # Obtain text rotation and alignment
        rotation, alignment = _get_label_rotation(angle, phase, tangential_labels=tangential_labels)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value + padding,
            s=label, 
            fontsize=def_label_fontsize,
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        )
    return None

def _add_group_text(angles, groups, group_cts, ax, pad_ct=0, n_plot=50):
    int_offset = 0
    for group, sz in zip(groups, group_cts):
        # add line below bars
        x1 = np.linspace(
                angles[int_offset + pad_ct], 
                angles[int_offset + sz + pad_ct],
                num=n_plot)
        ax.plot(x1, [-5]*n_plot, color="#333333")

        # add text to indicate group
        ax.text(
                np.mean(x1), -20, group, color="#333333",
                fontsize=def_label_fontsize, fontweight="bold",
                ha="center", va="center"
                )
    return None

def _get_label_set( varnames ):
    label_set = [None]*len(varnames)
    for i,var in enumerate(varnames):
        if "Convergent" in var:
            label = "Convergent"
        elif "Divergent" in var:
            label = "Divergent"
        else:
            label = var

        label_set[i] = label

    return label_set

def _get_label_df( series, label_name="label" ):
    varnames = series.index.tolist()
    label_set = _get_label_set( varnames )
    label_df = pd.concat( (series, 
                           pd.Series(
                               data=label_set,
                               index=series.index,
                               name=label_name)),
                           axis=1)
    return label_df


def _get_titles( chi2_result, short=True ):
    pair_name = chi2_result["obs_type"].replace("_XY_"," ").replace("sym","Symmetric Grid of ")[:-1] + " pairs"
    suptitle_stats, axtitle_stats = _get_title_stats( chi2_result )

    suptitle = f"{pair_name}\nIndependence test: {suptitle_stats}"
    if short:
        axtitles = axtitle_stats
    else:
        axtitles = [f"homogeneity: {ax_stats}" for ax_stats in axtitle_stats]
    return suptitle, axtitles


def _get_title_stats( chi2_result, debug=False ):
    statistics = list(zip(*chi2_result["statistics"]))
    chi2_indpt, pval_indpt = statistics[0]
    try:
        stat_pairs = list(zip(*statistics[2]))
    except IndexError as err:
        stat_pairs = list(zip(*statistics[-1]))
        if debug:
            print(f"(zipped) chi2 statstics output: \n{statistics}")
            print(f"Encountered error accessing 'statistics': \n{err}")
            print(f"Full results context: \n{chi2_result}")
            print("Exiting."); exit()

    suptitle_stats = _get_chi2_str(chi2_indpt, pval_indpt)
    axtitle_stats = [ _get_chi2_str(pair[0], pair[1]) for pair in stat_pairs ]
    return suptitle_stats, axtitle_stats

def _get_chi2_str(chi2, pval):
    chi2_str = f"$(\\chi^2, p) = ({'%.3g' % chi2}, {'%.2e' % pval})$"

    if pval < 0.05:
        n = min(3, int(np.abs(np.log10(pval*2))))
        chi2_str += "*"*n
    return chi2_str
