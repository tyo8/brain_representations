import os
import sys
import json
import argparse
import numpy as np
import permtest_utils as putils

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD

def_nulldir = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing"

## Primary functions
################################################################################################################
def get_permset_dists(barsX_fpath, barsY_fpath, 
        nulldir=def_nulldir, permtype="subject",
        persistence_type="diff", homdim=1, q=2, p=2,
        match_perms=True, verbose=True, debug=True):


    barsX = dgmD._get_bars(barsX_fpath, homdim=homdim)
    barsY = dgmD._get_bars(barsY_fpath, homdim=homdim)

    # compute standard Wasserstein distance between diagrams from X and Y
    data_summ = _summarize_data(barsX_fpath, barsY_fpath, debug=debug)

    if verbose:
        print(f"Computing ({p},{q})-Wasserstein distance between data-derived persistence H{homdim} diagrams...")

    data_summ = data_summ | putils.simple_distance(
            barsX, 
            barsY, 
            persistence_type=persistence_type,
            q=q, p=p,
            verbose=False
            )
    Wp_XY = data_summ["Wp_XY"]

    nullbarsX_pathlist = putils._permpaths_from_datapath(barsX_fpath, nulldir=nulldir, permtype=permtype, verbose=verbose, debug=debug)
    nullbarsX = [dgmD._get_bars(fpath, homdim=homdim) for fpath in nullbarsX_pathlist]

    nullbarsY_pathlist = putils._permpaths_from_datapath(barsY_fpath, nulldir=nulldir, permtype=permtype, verbose=verbose, debug=debug)
    nullbarsY = [dgmD._get_bars(fpath, homdim=homdim) for fpath in nullbarsY_pathlist]

    permlabels = [putils._parse_pathname(fpath, perm_pathtype=True, debug=False) for fpath in nullbarsX_pathlist]

    if verbose:
        print(f"X bars pulled from: \n{barsX_fpath}")
        print(f"Found {len(nullbarsX)} associated permuted nulls of type \"{permtype}\"")
        print(f"Y bars pulled from: \n{barsY_fpath}")
        print(f"Found {len(nullbarsY)} associated permuted nulls of type \"{permtype}\"")

    if debug:
        k=2
        ### debugging code ###
        print(f"Last {k} paths of permuted X nulls: \n{nullbarsX_pathlist[-k:]}")
        print(f"Last {k} paths of permuted Y nulls: \n{nullbarsY_pathlist[-k:]}")
        print(f"Restricting to intersection of first {min(len(nullbarsX), len(nullbarsY))} permutation sets")
        # assert len(nullbarsX)==len(nullbarsY), "Must pair permutations to compute distance distributions"
        ### debugging code ###
    else:
        if verbose:
            print(f"Restricting to intersection of first {min(len(nullbarsX), len(nullbarsY))} permutation sets")

    paired_nullbars = list(zip(nullbarsX, nullbarsY))

    perm_summ = {}
    perm_summ["datatype"] = "Null"
    perm_summ["permtype"] = permtype
    perm_summ["X_type"] = data_summ["X_type"]
    perm_summ["Y_type"] = data_summ["Y_type"]
    pairdist_summ = [data_summ] + [None]*len(paired_nullbars)

    if verbose:
        print(f"Computing Wasserstein distances between {len(paired_nullbars)} H{homdim} diagrams derived from permutation-matched null data...")

    for i, pair in enumerate(paired_nullbars):
        perm_dist = putils.simple_distance(
                pair[0], 
                pair[1],
                persistence_type=persistence_type,
                q=q, p=p,
                verbose=False
                )
        perm_dist["permlabel"] = permlabels[i]["permlabel"]
        pairdist_summ[i+1] = perm_summ | perm_dist 

    if debug:
        ### debugging code ###
        print("First entry of \'pairdist_summ\':", pairdist_summ[0])
        print(f"Last {k} entries of \'pairdist_summ\':")
        for entry in pairdist_summ[-k:]:
            print(entry)
        ### debugging code ###

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Datatype of \'X\':", data_summ["X_type"])
        print("Datatype of \'Y\':", data_summ["Y_type"])
        print("Projection cost of sending PD(X) to the empty diagram:", data_summ["PDX_diag"])
        print("Projection cost of sending PD(Y) to the empty diagram:", data_summ["PDY_diag"])
        print(f"Observed Wasserstein distance between X and Y: {Wp_XY}")
        print("Permutation type(s):", ', '.join(list(set([ dist["permtype"] for dist in pairdist_summ if dist["datatype"]=="Null" ]))))
        Wp_XYnull = np.array( [dist["Wp_XY"] for dist in pairdist_summ if dist["datatype"]=="Null"] )
        print(f"Summary of Wasserstein distance from data persistence diagrams to permuted-null persistence modules: \nmu={np.mean(Wp_XYnull)}, sigma={np.std(Wp_XYnull)}")
        print(f"Distribution of Wasserstein distances: \n{np.histogram(Wp_XYnull)}")
        print("\n")

    return pairdist_summ


def _get_outpath(data_summ, outdir=".", permtype="subject"):

    xlabel = data_summ["X_type"]
    ylabel = data_summ["Y_type"]

    outname = f"{xlabel}_vs_{ylabel}_null-{permtype}Perms.json"
    outpath = os.path.join(outdir, outname)

    return outpath


def _summarize_data(barsX_fpath, barsY_fpath, debug=True):
    
    xlabels = putils._parse_pathname(barsX_fpath, perm_pathtype=False, debug=debug)
    ylabels = putils._parse_pathname(barsY_fpath, perm_pathtype=False, debug=debug)

    data_summ = {}
    data_summ["datatype"] = "Data"
    data_summ["X_type"] = xlabels["modality"] + "_" + xlabels["feature"] + "_" + xlabels["metric"]
    data_summ["Y_type"] = ylabels["modality"] + "_" + ylabels["feature"] + "_" + ylabels["metric"]
    return data_summ
################################################################################################################


### FIX ARGPARSE!!!
################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Compute Wasserstein distance between persistence data derived from original data and their null-permuted derivatives"
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
        "-o", "--outdir",
        default='.', 
        type=str, 
        help="output directory"
    )
    parser.add_argument(
        "-D", "--nulldir",
        default=def_nulldir, 
        type=str, 
        help="directory containing null-permuted data"
    )
    parser.add_argument(
        "-d", "--dim", 
        default=1, 
        type=int, 
        help="homology dimension"
    )
    parser.add_argument(
        "-t", "--permtype", 
        default="subject", 
        type=str, 
        help="axis along which data is permuted (accepts either \'subject\' or \'feature\')"
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
        "-P", "--persistence_type", 
        default="diff", 
        type=str, 
        help="Either 'difference' or 'quotient' type measuremnt of persistence from birth/death values"
    )
    parser.add_argument(
        "-v", "--verbose", 
        default=False, 
        action="store_true",
        help="toggle verbose output"
    )
    args = parser.parse_args()

    putils._verify_distinct_spaces(args.barsX_fpath, args.barsY_fpath)

    if args.verbose:
        print(f"\n\nList of datasets pulled from: \n{args.barsX_fpath}")
        print(f"List of nullsets pulled from: \n{args.barsY_fpath}")
        print(f"Using permutations of type \'{args.permtype}\', from parent directory: \n{args.nulldir}")
        print(f"Computing (p,q)=({args.p},{args.q}) Wasserstein distance between diagrams of generators in H{args.dim}")
        print(f"Persistence values can be defined by either the \'difference\' or \'quotient\' of birth-death pairs: using \'{args.persistence_type}\'\n\n")

    pairdist_summ = get_permset_dists(
            args.barsX_fpath, 
            args.barsY_fpath,
            nulldir=args.nulldir,
            permtype=args.permtype,
            persistence_type=args.persistence_type, 
            homdim=args.dim, 
            q=args.q, 
            p=args.p,
            verbose=args.verbose,
            debug=True
            )
    outpath = _get_outpath(pairdist_summ[0], outdir=args.outdir, permtype=args.permtype)
    
    with open(outpath, 'w') as fout:
        json.dump(pairdist_summ, fout, indent=4)

    print(f"\n\nSaved to: \n{outpath}")
