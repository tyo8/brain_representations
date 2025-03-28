import os
import sys
import ast
import json
import argparse
import numpy as np
import permtest_utils as putils

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD

default_tagfile = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/subsampling/taglist100k_90p_famstruct.txt"

def bootstrap_distance(barsX_fpath, barsY_fpath, 
        tagfile=default_tagfile, count=1000, homdim=1, do_prev=True, do_dIM=True,
        use_affinity=False, persistence_type="diff", q=2, p=2, 
        verbose=True, debug=True):

    barsX = dgmD._get_bars(barsX_fpath, homdim=homdim)
    barsY = dgmD._get_bars(barsY_fpath, homdim=homdim)

    # compute standard Wasserstein distance between diagrams from X and Y
    Wp_XY = dgmD.weighted_Wasserstein_dist(barsX, barsY, wtfn_type=None, q=q, p=p, verbose=verbose)
    
    if do_prev:
        prevXpath = barsX_fpath.replace("bars_X.txt", f"prevalence_scores_dim{homdim}_n{count}_dict.txt") 
        prevYpath = barsY_fpath.replace("bars_X.txt", f"prevalence_scores_dim{homdim}_n{count}_dict.txt")

        if verbose:
            print(f"Pulling prevalence scores from: \nprevX_fpath: {prevXpath} \nprevY_fpath: {prevYpath}")
            
        prevX = np.loadtxt( prevXpath )
        prevY = np.loadtxt( prevYpath )

        if debug:
            print("X prevalence scores distirbuted as:")
            print(np.histogram(prevX))
            print("Y prevalence scores distirbuted as:")
            print(np.histogram(prevY))

        Wphat_XY = dgmD.weighted_Wasserstein_dist(barsX, barsY, w1=prevX, w2=prevY, wtfn_type="prevalence", q=q, p=p, verbose=verbose)
    else:
        Wphat_XY = np.nan 

    PDX_diag = dgmD.weighted_Wasserstein_dist(barsX, None, wtfn_type=None, q=q, p=p, verbose=verbose)
    PDY_diag = dgmD.weighted_Wasserstein_dist(None, barsY, wtfn_type=None, q=q, p=p, verbose=verbose)

    with open(tagfile,'r') as fin:
        taglist = [next(fin).split('\n')[0] for n in range(count)]
        prop = _get_subsample_prop(taglist[0])
        if debug:
            ### debugging code ###
            print(f"read {len(taglist)} tags from ${tagfile} after specifying a desired count of {count}.")
            ### debugging code ###

    # read lists of bars from list of filepaths
    barsXhat_flist = [os.path.join(os.path.dirname(barsX_fpath), 'phom_out', f"barsY_{tag}.txt") for tag in taglist]
    barsYhat_flist = [os.path.join(os.path.dirname(barsY_fpath), 'phom_out', f"barsY_{tag}.txt") for tag in taglist]

    barsXhat_list = [dgmD._get_bars(fpath) for fpath in barsXhat_flist]
    barsYhat_list = [dgmD._get_bars(fpath) for fpath in barsYhat_flist]

    if debug:
        ### debugging code ###
        putils._debug_bars_list(barsXhat_flist, barsXhat_list, name="X-hat_i")
        putils._debug_bars_list(barsYhat_flist, barsYhat_list, name="Y-hat_i")
        ### debugging code ###

    # pair persistence modules by subample tags
    persdgm_tagpairs = list(zip(barsXhat_list, barsYhat_list))

    # compute standard Wasserstein distance between co-tagged pairs
    Wp_XhatYhat_i = np.array([dgmD.weighted_Wasserstein_dist(diagram_pair[0], diagram_pair[1], wtfn_type=None, q=q, p=p, verbose=False) 
        for diagram_pair in persdgm_tagpairs])

    Wp_XXhat_i = np.array([dgmD.weighted_Wasserstein_dist(barsX, barsXhat, wtfn_type=None, q=q, p=p, verbose=False) for barsXhat in barsXhat_list])
    Wp_YYhat_i = np.array([dgmD.weighted_Wasserstein_dist(barsY, barsYhat, wtfn_type=None, q=q, p=p, verbose=False) for barsYhat in barsYhat_list])

    PDXhat_diag_i = np.array([dgmD.weighted_Wasserstein_dist(barsXhat, None, wtfn_type=None, q=q, p=p, verbose=False) for barsXhat in barsXhat_list])
    PDYhat_diag_i = np.array([dgmD.weighted_Wasserstein_dist(None, barsYhat, wtfn_type=None, q=q, p=p, verbose=False) for barsYhat in barsYhat_list])

    # cycle-registered distance (Omer & Bobrowski) between original diagram and its bootstrap (between matched cycles) 
    if do_dIM:
`       # read lists of cycle-registered bootstrap matches from list of filepaths
        Xmatch_fpaths = [os.path.join(os.path.dirname(barsX_fpath), 'matching', f"verbose_match_dim{homdim}_{tag}.txt") for tag in taglist]
        Ymatch_fpaths = [os.path.join(os.path.dirname(barsY_fpath), 'matching', f"verbose_match_dim{homdim}_{tag}.txt") for tag in taglist]

        # compute registered distance between all subset-generated persistence modules
        dIM_XXhat_i = get_registered_distances(Xmatch_fpaths, use_affinity=use_affinity, persistence_type=persistence_type, q=q, p=p)
        dIM_YYhat_i = get_registered_distances(Ymatch_fpaths, use_affinity=use_affinity, persistence_type=persistence_type, q=q, p=p)

        # compute set of composite bootstrap distances: 
        # matched module distance d(X,Xhat) + paired diagram distance Wp(Xhat, Yhat) + matched module distance d(Y,Yhat)
        Wphat0_XY_i = dIM_XXhat_i + Wp_XhatYhat_i + dIM_YYhat_i
    else:
        Wphat0_XY_i = Wp_XhatYhat_i + np.nan
        dIM_XXhat_i = Wp_XhatYhat_i + np.nan
        dIM_YYhat_i = Wp_XhatYhat_i + np.nan

    results_dict = {
            "Xname": barsX_fpath,
            "Yname": barsY_fpath,
            "Wp_XY": Wp_XY,
            "Wphat_XY": Wphat_XY, 
            "Wphat0_XY": Wphat0_XY, 
            "Wphat0_XY_i": Wphat0_XY_i, 
            "Wp_XhatYhat_i": Wp_XhatYhat_i, 
            "Mean Wp Approximation Difference": Wp_XY - np.mean(Wp_XhatYhat_i),
            "PDX_diag": PDX_diag, 
            "PDY_diag": PDY_diag, 
            "Wp_XXhat_i": Wp_XXhat_i, 
            "Wp_YYhat_i": Wp_YYhat_i, 
            "PDXhat_diag_i": PDXhat_diag_i, 
            "PDYhat_diag_i": PDYhat_diag_i,
            "dIM_XXhat_i": dIM_XXhat_i, 
            "dIM_YYhat_i": dIM_YYhat_i
            }

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Number of resamples:", count)
        print("Resample proportion:", prop)
        print("Standard Wasserstein distance between PD(X) and PD(Y):", Wp_XY)
        if do_prev:
            print("Prevalence-weighted Wasserstein distance between PD(X) and PD(Y):", Wphat_XY)
        print("")
        print(f"2-sigma interval around standard Wasserstein distance between PD(Xhat_i) and PD(Yhat_i): {np.mean(Wp_XhatYhat_i)} +/- {2*np.std(Wp_XhatYhat_i)}")
        print(f"2-sigma interval around standard Wasserstein distance between PD(X) and PD(Xhat_i) = {np.mean(Wp_XXhat_i)} +/- {2*np.std(Wp_XXhat_i)}")
        print(f"2-sigma interval around standard Wasserstein distance between PD(Y) and PD(Yhat_i) = {np.mean(Wp_YYhat_i)} +/- {2*np.std(Wp_YYhat_i)}")
        print("")
        print(f"2-sigma interval around \"bootstrapped Wasserstein distance\" between PD(X) and PD(Y): {np.mean(Wphat0_XY_i)} +/- {2*np.std(Wphat0_XY_i)}")
        print(f"2-sigma interval around matched diagram distance between PD(X) and PD(Xhat_i): {np.mean(dIM_XXhat_i)} +/- {2*np.std(dIM_XXhat_i)}")
        print(f"2-sigma interval around matched diagram distance between PD(Y) and PD(Yhat_i): {np.mean(dIM_YYhat_i)} +/- {2*np.std(dIM_YYhat_i)}")
        print("")
        print("Projection cost of sending PD(X) to the empty diagram:", PDX_diag)
        print(f"Projection cost (2-sigma interval) of sending PD(Xhat) to the empty diagram: {np.mean(PDXhat_diag_i)} +/- {2*np.std(PDXhat_diag_i)}")
        print("Projection cost of sending PD(Y) to the empty diagram:", PDY_diag)
        print(f"Projection cost (2-sigma interval) of sending PD(Yhat) to the empty diagram: {np.mean(PDYhat_diag_i)} +/- {2*np.std(PDYhat_diag_i)}")

    return results_dict


def get_registered_distances(vbmatches_filelist, use_affinity=True, 
        persistence_type="diff", q=2, p=2):
    vbmatches = [None]*len(vbmatches_filelist)
    for i,fpath in enumerate(vbmatches_filelist):
        with open(fpath, 'r') as fin:
            vbmatches[i] = ast.literal_eval(fin.read())

    registered_dists = np.array([
        dgmD.module_distance(
        vbmatch,
        use_affinity=use_affinity, 
        persistence_type=persistence_type,
        q=q,
        p=p) for vbmatch in vbmatches])
    return registered_dists

def _get_subsample_prop(tag):
    from generate_subindex import tag_to_subidx
    subidx = tag_to_subidx(tag)
    prop = len(subidx)/(1 + max(subidx))
    return prop


################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-x",
        "--barsX_fpath",
        default=None,
        type=str,
        help="filepath to bars of first persistence diagram"
    )
    parser.add_argument(
        "-y",
        "--barsY_fpath",
        default=None,
        type=str,
        help="filepath to bars of second persistence diagram"
    )
    parser.add_argument(
        "-t", "--tagfile", 
        default=default_tagfile, 
        type=str, 
        help="output filepath to persistence diagram image"
    )
    parser.add_argument(
        "-d", "--dim", 
        default=1, 
        type=int, 
        help="homology dimension"
    )
    parser.add_argument(
        "-n", "--count", 
        default=100, 
        type=int, 
        help="output filepath to persistence diagram image"
    )
    parser.add_argument(
        "-a", "--use_affinity", 
        default=True, 
        type=bool, 
        help="toggle inclusion of affinity score information when calculating between-bootstrapped persistence module distance"
    )
    parser.add_argument(
        "-I", "--do_dIM", 
        default=True, 
        type=bool, 
        help="toggle inclusion of cycle-registered distance when calculating between-bootstrapped persistence module distance"
    )
    parser.add_argument(
        "-p", "--p", 
        default=2, 
        type=int, 
        help="norm power of Wasserstein distance (positive integer)"
    )
    parser.add_argument(
        "-q", "--q", 
        default=2, 
        type=int, 
        help="norm power of diagram (i.e., Euclidean) distance (np.inf or positive integer)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        default=False, 
        action="store_true",
        help="toggle verbose output"
    )
    parser.add_argument(
        "-o", "--outpath", 
        default=None, 
        type=str, 
        help="output filepath to saved output"
    )
    args = parser.parse_args()

    putils._verify_distinct_spaces(args.barsX_fpath, args.barsY_fpath)

    if args.verbose:
        print("")
        print(f"Computing bootstraps through {args.count} resamples from taglist: \n{args.tagfile}")
        print(f"Incorporating affinity scores in between-bootstrapped persistence module distance computation: {args.use_affinity}")
        print(f"Incorporating cycle-matched distance in between-bootstrapped persistence module distance computation: {args.do_dIM}")
        print("")

    output = bootstrap_distance(
            args.barsX_fpath, 
            args.barsY_fpath, 
            tagfile=args.tagfile, 
            count=args.count, 
            homdim=args.dim, 
            use_affinity=args.use_affinity, 
            do_dIM=args.do_dIM, 
            persistence_type="diff", 
            q=args.q, 
            p=args.p, 
            verbose=args.verbose
            )

    with open(args.outpath, 'w') as fout:
        json.dump(output, fout, indent=4)
