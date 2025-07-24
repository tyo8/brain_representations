import os
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import figutils as futils
import figstats as fstats
from matplotlib import pyplot as plt

def make(value_set, args):
    fig, outname, summary_stackplot_df = do_stackplot(value_set, alpha=args.alpha)
    # summary_stackplot_df.to_csv('value_set/summary_stackplot_df.csv')
    outpath = os.path.join(args.output_dir, outname)
    futils._write_img(fig, outpath, fig_size=args.fig_size)
    return summary_stackplot_df

def do_stackplot(
        value_set,
        alpha=0.05,
        epsilon=2e-3,
        verbose=True,
        debug=True
        ):

    if alpha is None:
        alpha = epsilon

    Brain_Representation = copy.copy(value_set["Y_name"][0])
    Brain_Representation[0] = copy.copy(value_set["X_name"][0][1])

    left_pvalvar = [var for var in list(value_set.keys()) if 'left-pval' in var][0]
    right_pvalvar = [var for var in list(value_set.keys()) if 'right-pval' in var][0]

    convergent_count = np.sum(value_set[left_pvalvar] < alpha, axis=0) - 1      # '- 1' to remove self-comparison from convergence count
    divergent_count = np.sum(value_set[right_pvalvar] < alpha, axis=0)
    incomparable_count = len(Brain_Representation) - 1 - convergent_count - divergent_count    # '- 1' to remove self-comparison from total count

    stackplot_df = pd.DataFrame( {
        "Brain_Representation":Brain_Representation, 
        "Convergent":convergent_count, 
        "Divergent":divergent_count, 
        "Incomparable":incomparable_count} )
    stackplot_df[["Parcellation","Feature","Metric"]] = stackplot_df["Brain_Representation"].str.split('_', n=2, expand=True)
    stackplot_df.sort_values("Feature", inplace=True)
    stackplot_df.Brain_Representation = stackplot_df.Parcellation + "_" + stackplot_df.Feature
    stackplot_df.drop( columns = ["Metric"], inplace=True )
    stackplot_df["Feature"] = [ 'NM' if 'NM' in var else var for var in stackplot_df["Feature"].to_list() ] 

    varnames = ["Brain_Representation", "Parcellation", "Feature"]

    fig, axes = plt.subplots(len(varnames))
    sns.set_theme()
    for i,var in enumerate(varnames):
        ax = axes[i]
        dropvars = copy.copy(varnames)
        dropvars.remove(var)

        var_df = fstats._contract_df( stackplot_df.drop(columns=dropvars), agg_col=var )

        fig,ax = vert_stackplot(
                stackplot_df= var_df,
                count_colname=var,
                fig=fig, ax=ax
                )

        ax.set_title(f"Cumulative Comparsion Types by {var}\n(significance = {alpha})")
        ax.tick_params(axis='y', labelrotation=30)

    fig.tight_layout

    outname = f"cumulative-sig-counts_alpha{alpha}.png".replace('0.','')

    return fig, outname, stackplot_df


def vert_stackplot(
        stackplot_df=None, count_colname=None, 
        counts_by_name=None, category_names=None,
        fig=None, ax=None
        ):
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
    if stackplot_df is None:
        assert counts_by_name is not None, "varible 'counts_by_name' must be nontrivial if input dataframe is not given"
        assert category_names is not None, "varible 'category_names' must be nontrivial if input dataframe is not given"
        labels = list(counts_by_name.keys())
        data = np.array(list(counts_by_name.values()))
    else:
        category_names = stackplot_df.columns.values.tolist()
        if count_colname is None:
            labels = stackplot_df.index.tolist()
        else:
            labels = stackplot_df[count_colname].values.tolist()
            category_names.remove(count_colname)
        data = stackplot_df[category_names].values.astype(float)


    data_cum = data.cumsum(axis=1)
    
    colors = sns.color_palette(as_cmap=True)

    if (ax is None) or (fig is None):
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


