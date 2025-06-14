import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import figutils as futils
from matplotlib import pyplot as plt

def make(value_set, args):
    fig, outname = do_stackplot(value_set, alpha=args.alpha)
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=(12,6))
    return None

def do_stackplot(
        value_set,
        alpha=0.05,
        epsilon=2e-3,
        verbose=True,
        debug=True
        ):

    if alpha is None:
        alpha = epsilon

    Names = copy.copy(value_set["Y_name"][0])
    Names[0] = copy.copy(value_set["X_name"][0][1])

    left_pvalvar = [var for var in list(value_set.keys()) if 'fdr-left-pval' in var][0]
    right_pvalvar = [var for var in list(value_set.keys()) if 'fdr-right-pval' in var][0]

    convergent_count = np.sum(value_set[left_pvalvar] < alpha, axis=0) - 1      # '- 1' to remove self-comparison from convergence count
    divergent_count = np.sum(value_set[right_pvalvar] < alpha, axis=0)
    incomparable_count = len(Names) - 1 - convergent_count - divergent_count    # '- 1' to remove self-comparison from total count

    df = pd.DataFrame( {
        "Names":Names, 
        "Convergent":convergent_count, 
        "Divergent":divergent_count, 
        "Incomparable":incomparable_count} )
    df[["Modality","Feature","Metric"]] = df["Names"].str.split('_', n=2, expand=True)
    df.sort_values("Feature", inplace=True)
    df.Names = df.Modality + "_" + df.Feature
    df.drop( columns = ["Modality", "Feature", "Metric"], inplace=True )

    category_names = df.columns.values[1:]
    X = df["Names"].to_list()
    Y = df[category_names].to_numpy()
    counts_by_name = {}
    for i,name in enumerate(X):
        counts_by_name[name] = Y[i]

    sns.set_theme()

    fig,ax = vert_stackplot(counts_by_name, category_names)

    ax.set_title("Cumulative Comparsion Types by Brain Representation")
    ax.tick_params(axis='y', labelrotation=30)
    fig.tight_layout

    outname = f"per-BR_cumulative-sig-counts_alpha{alpha}.png"

    return fig, outname


def vert_stackplot(counts_by_name, category_names):
    """
    Parameters
    ----------
    counts_by_name : dict
        A mapping from label category_names to a list of counts per category.
        It is assumed all lists contain the same number of entries and that
        the sum of entries matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(counts_by_name.keys())
    data = np.array(list(counts_by_name.values()))

    data_cum = data.cumsum(axis=1)
    
    colors = sns.color_palette(as_cmap=True)

    fig, ax = plt.subplots()

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=1,
                        label=colname, color=color)

        text_color = 'lightgrey'
        ax.bar_label(rects, label_type='center', color=text_color)

    ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')
    return fig, ax
