import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

palette_list = ['RdPu_r', 'PuBuGn_r', 'YlOrBr_r', 'Blues', 'GnBu_r', 'Purples_r']
# palette_list = ['RdPu', 'PuBuGn', 'YlOrBr', 'Blues_r', 'GnBu', 'Purples']
def_outdir = "/home/tyo/Documents/Personomics_Lab/BBRP_writeup/resubmission/scratch/methods_figs_parts"

def subj_shuf_vecs(N=3, n=20, display=True, write=True, figsize=(7,7), outdir=def_outdir):
    fig,axes = plt.subplots(1, N, sharey=True)
    fig.set_size_inches(figsize)
    cmap_list = [None]*N

    vec = np.array(list(range(n))).reshape(-1,1)

    for i in range(N):
        # cmap = mpl.colors.ListedColormap( sns.color_palette( palette=palette_list[i], desat=1, n_colors=n ))
        cmap = sns.color_palette(palette=palette_list[i], as_cmap=True)
        sns.heatmap( vec, cmap=cmap, ax=axes[i], cbar=False )
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    if display:
        plt.show()

    if write:
        outpath = os.path.join(outdir, "subj_shufs.png")
        fig.savefig(outpath, dpi=600)

    return None


def feat_shuf_vecs(n=20, display=True, write=True, figsize=(7,7), outdir=def_outdir, seed=0):
    from sympy.combinatorics import Permutation

    rng = np.random.default_rng(seed)
    fig,axes = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(figsize)

    vec = np.array(list(range(n))).reshape(-1,1)
    p_vec = rng.permuted(vec)
    
    cmap = sns.color_palette(palette='Spectral', as_cmap=True)
    # cmap = sns.color_palette(palette='Set3', as_cmap=True)
    # cmap = sns.color_palette(palette='husl', as_cmap=True)
    for i,v in enumerate([vec, p_vec]):
        sns.heatmap( v, cmap=cmap, ax=axes[i], cbar=False )

    for ax in axes:
        ax.set_xticks([]), 
        ax.set_yticks([])

    trpns = Permutation(p_vec.flatten()).transpositions()

    print(f"List of Transpositions: \n{trpns}")

    if display:
        plt.show()

    if write:
        outpath = os.path.join(outdir, "feat_shufs.png")
        fig.savefig(outpath, dpi=600)
        t_outpath = os.path.join(outdir, "transpositions.txt")
        with open(t_outpath,'w') as fout:
            for t in trpns:
                fout.write(f"{t}\n")

    return None

#######################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="create components of data permutation schematic"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=def_outdir,
        help="output directory"
    )
    parser.add_argument(
        "-w",
        "--write",
        default=False,
        action="store_true",
        help="write plots to .png"
    )
    parser.add_argument(
        "-d",
        "--display",
        default=False,
        action="store_true",
        help="display plots"
    )
    parser.add_argument(
        "-N",
        "--number_subjs",
        type=int,
        default=3,
        help="number of displayed vectors"
    )
    parser.add_argument(
        "-n",
        "--number_dims",
        type=int,
        default=13,
        help="number of vector \"components\""
    )
    args = parser.parse_args()

    subj_shuf_vecs(
            N = args.number_subjs, 
            n = args.number_dims, 
            display = args.display, 
            write = args.write, 
            outdir = args.outdir
            )


    feat_shuf_vecs(
            n = args.number_dims, 
            display = args.display, 
            write = args.write, 
            outdir = args.outdir,
            seed = 196883
            )
