import os
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import figstats as fstats
from matplotlib import pyplot as plt

sns.set_theme()
golden = (1 + 5**0.5)/2

def pair_distribution_curve( x=np.linspace(0,10,1000), ctype="bumpy"):
    if ctype == "curt":
        f_t = lambda t: np.exp(-(t-3)**2) + 2 * np.exp(-((t-6)**2)/4) * (1-np.exp((-1/(t-5.)**2)))
    elif ctype == "bumpy":
        f_t = lambda t: np.exp(-(t-3)**2) + 2 * np.exp(-((t-6)**2)/4)
    elif ctype == "simple":
        f_t = lambda t: np.exp(-(t-3)**2)
    else:
        f_t = lambda t: 1/( 1 + (t-3)**2 )

    pdf = f_t(x) / sum(f_t(x))*np.mean(np.diff(x))

    curve = np.array([x, pdf])

    return pdf, curve


def pairplot(curve, ax, alpha=0.05):
    x,y = curve[0], curve[1]
    ax.plot(x, y, color='k', label=r"$D_{ij}^{(\sigma)} \sim f(d)$", linewidth=2)

    left_lim, right_lim = epdf_lims(curve, alpha=alpha)
    ax.axvline(left_lim, color="gray", ls=":", label=r"$\int f = \alpha$")
    ax.axvline(right_lim, color="gray", ls="--", label=r"$\int f = 1-\alpha$")

    ax.fill_between(x, 0, y, where=(x < left_lim), label="Convergent", 
                    alpha=0.8, hatch='\\\\\\\\', hatch_linewidth=1)
    ax.fill_between(x, 0, y, where=(x > right_lim), label="Divergent" , 
                    alpha=0.8, hatch='////', hatch_linewidth=1) 
    ax.fill_between(x, 0, y, where=((x > left_lim) & (x < right_lim)), label="Incomparable", 
                    alpha=0.8, hatch='----', hatch_linewidth=1) 
    ax.legend(loc="upper left")

    ax.set(xticklabels = [])
    ax.set(yticklabels = [])
    ax.set_xlabel(r"Wasserstein Distance $d$")
    ax.set_ylabel(r"Permutation Frequency $f$")
    ax.set_title("Pairwise Difference Filtering")
    return None

# give the left and right input values with respective tail probabilities at most 'alpha'

def epdf_lims( epdf, alpha=0.05, debug=False ):
    x,y = epdf[0], epdf[1]
    ecdf = np.cumsum(y) / sum(y)

    if debug:
        fig0,ax0 = plt.subplots()
        ax0.plot(x, ecdf)

    left_lim = x[max( np.argwhere( ecdf < alpha ) )]
    right_lim = x[min( np.argwhere( ecdf > 1-alpha ) )]
    return left_lim, right_lim



def ROC_curve( alpha=0.05, debug=True ):
    eps_set = np.linspace(0,3,8001)
    m_alpha = lambda m: 1 + 1/2 * ( scipy.special.erf( -m )  - scipy.special.erf( m ) )

    sep = eps_set[ np.argmin( np.abs(m_alpha(eps_set)/2 - alpha) ) ]
    if debug:
        print(f"Midpoint of Gaussians: \tx={sep}")

    null_dist = np.random.randn(10000)
    true_dist = np.random.randn(10000) + 2*sep

    roc_curve, auc = fstats.get_roc(true_dist, null_dist)
    if debug:
        print("\tEmpirical ROC alpha:", 1-auc, "\n\tRequested ROC alpha:", alpha)
    return roc_curve, auc, [null_dist, true_dist, sep]



def inlay_figure( truedist, nulldist, midpoint, 
                 ctrue = sns.color_palette()[4],
                 cnull = sns.color_palette()[9],
                 alpha=0.05, figsize=(8,8), debug=False ):
    fig, ax = plt.subplots( figsize=figsize )

    df = pd.concat( [
        pd.DataFrame({"Values": truedist, "Distribution":"Bootstrap"}),
        pd.DataFrame({"Values": nulldist, "Distribution":"Null-derived"})
        ], ignore_index=True)

    sns.kdeplot(data=df, x="Values", hue="Distribution", palette=[ctrue, cnull], ax=ax)

    xtrue, ytrue = ax.lines[0].get_data()
    xnull, ynull = ax.lines[1].get_data()
    x_all = np.linspace(min(min(xtrue),min(xnull)), max(max(xtrue),max(xnull)), len(xnull) + len(xtrue))
    y_low = np.array([ min([query(xtrue, ytrue, x, debug=debug), query(xnull, ynull, x)]) for x in x_all ])

    if debug:
        print(f"y_low has shape {y_low.shape}")
        print(f"original curves: \n\t{ytrue.shape} \n\t{ynull.shape}")
        fig0,ax0 = plt.subplots()
        ax0.plot(xtrue, ytrue)
        ax0.plot(xnull, ynull)
        ax0.plot(x_all, y_low)
        plt.show()

    ax.fill_between(xnull, 0, ynull, where=((xnull > midpoint/2) & (xnull < midpoint)), 
                    color=ctrue, label="True positives (Bootstraps)" )
    ax.fill_between(x_all, 0, y_low, where=(x_all > midpoint/2), 
                    color=cnull, label="False positives (Nulls)" )
    ax.set(xticklabels = [])
    ax.set(yticklabels = [])
    ax.legend(loc="upper right")
    ax.set_xlabel(r"Wasserstein distance $d$")
    # ax.tick_params(bottom=False)
    # ax.tick_params(left=False)
    return fig


def query(support, image, query_pt, debug=False):
    if all(query_pt < support) or all(query_pt > support):
        return 0
    idx = np.max(np.where( query_pt >= support ))
    val = image[idx]
    if debug:
        print(f"query point {query_pt} has left edge at index {idx}, with value {val}")

    return val


def roc_plot(curve, ax):
    colors = sns.color_palette()
    ax.plot(curve[0], curve[1], label="ROC curve")
    ax.fill_between(
            curve[0], curve[1], 1, 
            color=colors[6],
            label=r"$1 \geq \text{AUC} \geq 1-\alpha$"
            )

    ax.set_xlabel("True Positive Rate")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("Single BrainRep Filtering")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.legend(loc="lower right")
    return None


def schematic_figure( alpha=0.05, figsize=(18,9/golden) ):
    sns.set_style("dark")
    fig, axes = plt.subplots( 
                             nrows=1, ncols=2, 
                             figsize=figsize,
                             gridspec_kw = {"width_ratios": [1,golden]},
                             )

    ax_roc = axes[0]
    ax_pair = axes[1]

    roc_curve, auc, [truedist, nulldist, m] = ROC_curve( alpha=alpha )
    roc_plot(roc_curve, ax_roc)
    inlay_figure(truedist, nulldist, m, alpha=alpha)

    pairpdf, pairpdf_curve = pair_distribution_curve()
    pairplot(pairpdf_curve, ax_pair, alpha=alpha)

    return fig


if __name__=="__main__":
    alpha = 0.05
    main_fig = schematic_figure(alpha=alpha)
    plt.show()
