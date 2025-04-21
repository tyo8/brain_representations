import os
import sys
import ast
import json
import argparse
import numpy as np
import pandas as pd
import permtest_utils as putils

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD

default_tagfile = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/subsampling/taglist100k_90p_famstruct.txt"


def bootstrap_distance(barsX_fpath, nametype="X",
        tagfile=default_tagfile, count=1000, homdim=1, do_prev=False, do_dIM=True,
        match_only=True, use_affinity=False, persistence_type="diff", q=2, p=2, 
        verbose=True, debug=True):

    barsX = dgmD._get_bars(barsX_fpath, homdim=homdim)

    if do_prev:
        prevXpath = barsX_fpath.replace("bars_X.txt", f"prevalence_scores_dim{homdim}_n{count}_dict.txt") 

        if verbose:
            print(f"Pulling prevalence scores from: \nprevX_fpath: {prevXpath}")
            
        prevX = np.loadtxt( prevXpath )

        if debug:
            print("X prevalence scores distirbuted as:")
            print(np.histogram(prevX))
    else:
        prevX = None

    PDX_diag = dgmD.weighted_Wasserstein_dist(barsX, None, wtfn_type=None, q=q, p=p, verbose=verbose)

    with open(tagfile,'r') as fin:
        taglist = [next(fin).split('\n')[0] for n in range(count)]
        prop = _get_subsample_prop(taglist[0])
        if debug:
            ### debugging code ###
            print(f"read {len(taglist)} tags from ${tagfile} after specifying a desired count of {count}.")
            print(f"the (empricial) subsampling proportion of the first tag in the list approx. {prop}")
            ### debugging code ###
    taglist.sort()

    if match_only:
        # read lists of cycle-registered bootstrap matches from list of filepaths
        barsXhat_flist = [os.path.join(os.path.dirname(barsX_fpath), 'matching', f"verbose_match_dim{homdim}_{tag}.txt") for tag in taglist]
        barsXhat_list = [dgmD._get_matched_bars(fpath) for fpath in barsXhat_flist]
    else:
        # read lists of bars from list of filepaths
        barsXhat_flist = [os.path.join(os.path.dirname(barsX_fpath), 'phom_out', f"barsY_{tag}.txt") for tag in taglist]
        barsXhat_list = [dgmD._get_bars(fpath) for fpath in barsXhat_flist]

    if debug:
        ### debugging code ###
        putils._debug_bars_list(barsXhat_flist, barsXhat_list, name="X-hat_i")
        ### debugging code ###

    Wp_XXhat_i = np.array([dgmD.weighted_Wasserstein_dist(barsX, barsXhat, wtfn_type=None, q=q, p=p, verbose=False) for barsXhat in barsXhat_list])

    PDXhat_diag_i = np.array([dgmD.weighted_Wasserstein_dist(barsXhat, None, wtfn_type=None, q=q, p=p, verbose=False) for barsXhat in barsXhat_list])

    if do_dIM:
        # cycle-registered distance (Omer & Bobrowski) between original diagram and its bootstrap (between matched cycles) 
        # compute registered distance between all subset-generated persistence modules
        dIM_XXhat_i = get_registered_distances(Xmatch_fpaths, use_affinity=use_affinity, persistence_type=persistence_type, q=q, p=p)
    else:
        dIM_XXhat_i = None

    X_name = os.path.basename(os.path.dirname(barsX_fpath)).replace("phom_data_","").replace("_dists","")

    bsdict_X = {
            "datatype": "Subsamp",
            "X_name": X_name,
            "X_path": barsX_fpath,
            "taglist": taglist,
            "PDX_diag": PDX_diag, 
            "Wp_XXhat_i": Wp_XXhat_i, 
            "PDXhat_diag_i": PDXhat_diag_i, 
            "dIM_XXhat_i": dIM_XXhat_i
            }

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Number of resamples:", count)
        print("Resample proportion:", prop)
        print(f"2-sigma interval around standard Wasserstein distance between PD(X) and PD(Xhat_i) = {np.mean(Wp_XXhat_i)} +/- {2*np.std(Wp_XXhat_i)}")
        print("")
        if do_dIM:
            print(f"2-sigma interval around matched diagram distance between PD(X) and PD(Xhat_i): {np.mean(dIM_XXhat_i)} +/- {2*np.std(dIM_XXhat_i)}")
        print("")
        print("Projection cost of sending PD(X) to the empty diagram:", PDX_diag)
        print(f"Projection cost (2-sigma interval) of sending PD(Xhat) to the empty diagram: {np.mean(PDXhat_diag_i)} +/- {2*np.std(PDXhat_diag_i)}")

    xvarnames = list(bsdict_X.keys())
    yvarnames = [ name.replace("X", nametype) for name in xvarnames ]
    for i,name in enumerate(xvarnames):
        bsdict_X[yvarnames[i]] = bsdict_X.pop(name)

    return bsdict_X, [prevX, dIM_XXhat_i, barsXhat_list]

def pairstrap_distance(barsX_fpath, barsY_fpath, 
        tagfile=default_tagfile, count=1000, homdim=1, do_prev=True, do_dIM=True,
        use_affinity=False, persistence_type="diff", q=2, p=2, 
        verbose=True, debug=False):

    barsX = dgmD._get_bars(barsX_fpath, homdim=homdim)
    barsY = dgmD._get_bars(barsY_fpath, homdim=homdim)

    # compute standard Wasserstein distance between diagrams from X and Y
    Wp_XY = dgmD.weighted_Wasserstein_dist(barsX, barsY, wtfn_type=None, q=q, p=p, verbose=verbose)
    
    bsdict_X, [prevX, dIM_XXhat_i, barsXhat_list] = bootstrap_distance(barsX_fpath, nametype="X",
                                                                tagfile=tagfile, 
                                                                count=count, 
                                                                homdim=homdim, 
                                                                do_prev=do_prev, 
                                                                do_dIM=do_dIM,
                                                                use_affinity=use_affinity, 
                                                                persistence_type=persistence_type, 
                                                                q=q, p=p, 
                                                                verbose=verbose)

    bsdict_Y, [prevY, dIM_YYhat_i, barsYhat_list] = bootstrap_distance(barsY_fpath, nametype="Y",
                                                                tagfile=tagfile, 
                                                                count=count, 
                                                                homdim=homdim, 
                                                                do_prev=do_prev, 
                                                                do_dIM=do_dIM,
                                                                use_affinity=use_affinity, 
                                                                persistence_type=persistence_type, 
                                                                q=q, p=p, 
                                                                verbose=verbose)

    if do_prev:
        if debug:
            print("X prevalence scores distirbuted as:")
            print(np.histogram(prevX))
            print("Y prevalence scores distirbuted as:")
            print(np.histogram(prevY))

        Wphat_XY = dgmD.weighted_Wasserstein_dist(barsX, barsY, w1=prevX, w2=prevY, wtfn_type="prevalence", q=q, p=p, verbose=verbose)
    else:
        Wphat_XY = None

    # pair persistence modules by subample tags
    persdgm_tagpairs = list(zip(barsXhat_list, barsYhat_list))

    # compute standard Wasserstein distance between co-tagged pairs
    Wp_XhatYhat_i = np.array([dgmD.weighted_Wasserstein_dist(diagram_pair[0], diagram_pair[1], wtfn_type=None, q=q, p=p, verbose=False) 
        for diagram_pair in persdgm_tagpairs])

    # cycle-registered distance (Omer & Bobrowski) between original diagram and its bootstrap (between matched cycles) 
    if do_dIM:
        # compute set of composite bootstrap distances: 
        # matched module distance d(X,Xhat) + paired diagram distance Wp(Xhat, Yhat) + matched module distance d(Y,Yhat)
        Wphat0_XY_i = dIM_XXhat_i + Wp_XhatYhat_i + dIM_YYhat_i
    else:
        Wphat0_XY_i = None

    new_results_dict = {
            "Wp_XY": Wp_XY,
            "Wphat_XY": Wphat_XY, 
            "Wphat0_XY_i": Wphat0_XY_i, 
            "Wp_XhatYhat_i": Wp_XhatYhat_i, 
            "Mean Wp Approximation Difference": Wp_XY - np.mean(Wp_XhatYhat_i)
            }

    results_dict = new_results_dict | bsdict_X | bsdict_Y

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Number of resamples:", count)
        print("Standard Wasserstein distance between PD(X) and PD(Y):", Wp_XY)
        if do_prev:
            print("Prevalence-weighted Wasserstein distance between PD(X) and PD(Y):", Wphat_XY)
        print("")
        print(f"2-sigma interval around standard Wasserstein distance between PD(Xhat_i) and PD(Yhat_i): {np.mean(Wp_XhatYhat_i)} +/- {2*np.std(Wp_XhatYhat_i)}")
        if do_dIM:
            print(f"2-sigma interval around \"bootstrapped Wasserstein distance\" between PD(X) and PD(Y): {np.mean(Wphat0_XY_i)} +/- {2*np.std(Wphat0_XY_i)}")

    return results_dict, bsdict_X


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
# parses input, saves results
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of results from topological bootstrap"
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
        help="results filepath to persistence diagram image"
    )
    parser.add_argument(
        "-d", "--dim", 
        default=1, 
        type=int, 
        help="homology dimension"
    )
    parser.add_argument(
        "-n", "--count", 
        default=1000, 
        type=int, 
        help="results filepath to persistence diagram image"
    )
    parser.add_argument(
        "-M", "--match_only", 
        default=False, 
        action="store_True",
        help="disable 'free Wasserstein' computation (we are not allowed to use bars in bootstrapped Y with no match in X)"
    )
    parser.add_argument(
        "-a", "--use_affinity", 
        default=False, 
        action="store_true", 
        help="toggle inclusion of affinity score information when calculating between-bootstrapped persistence module distance"
    )
    parser.add_argument(
        "-I", "--do_dIM", 
        default=False, 
        action="store_true", 
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
        help="toggle verbose results"
    )
    parser.add_argument(
        "-o", "--outpath", 
        default=None, 
        type=str, 
        help="results filepath to saved results"
    )
    parser.add_argument(
        "-O", "--outpath2", 
        default=None, 
        type=str, 
        help="results filepath to saved results"
    )
    args = parser.parse_args()
    args = parser.parse_args()

    putils._verify_distinct_spaces(args.barsX_fpath, args.barsY_fpath)

    if args.verbose:
        print("")
        print(f"Computing bootstraps through {args.count} resamples from taglist: \n{args.tagfile}")
        print(f"Incorporating affinity scores in between-bootstrapped persistence module distance computation: {args.use_affinity}")
        print(f"Incorporating cycle-matched distance in between-bootstrapped persistence module distance computation: {args.do_dIM}")
        print("")

    results, Xresults = pairstrap_distance(
            args.barsX_fpath, 
            args.barsY_fpath, 
            tagfile=args.tagfile, 
            count=args.count, 
            homdim=args.dim,
            match_only=args.match_only,
            use_affinity=args.use_affinity, 
            do_prev=args.use_affinity, 
            do_dIM=args.do_dIM, 
            persistence_type="diff", 
            q=args.q, 
            p=args.p, 
            verbose=args.verbose
            )

    ## convert these to dataframe for easier write-out
    res_df = pd.DataFrame(data = results)
    res_df.to_csv(args.outpath.replace("_ztrans","-ztrans"))

    Xres_df = pd.DataFrame(data = Xresults)
    Xres_df.to_csv(args.outpath2.replace("_ztrans","-ztrans"))
