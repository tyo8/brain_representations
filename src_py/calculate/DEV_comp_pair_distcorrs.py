import os
import sys
import argparse
import numpy as np
import pandas as pd
import permtest_utils as putils

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD

def_nulldir = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing"

## Primary functions
################################################################################################################
def get_permset_distcorrs(dX_fpath, dY_fpath, 
        nulldir=def_nulldir, permtype="subject",
        persistence_type="diff", homdim=1, q=2, p=2,
        match_perms=True, verbose=True, debug=True):


    dX = np.loadtxt(dX_fpath)
    dY = np.loadtxt(dY_fpath)

    # compute standard Wasserstein distance between diagrams from X and Y
    data_summ = _summarize_data(dX_fpath, dY_fpath, debug=debug)

    if verbose:
        print(f"Computing distance correlation between raw subject-by-subject distance matrices...")


    #################################### REPLACE BY DISTANCE CORRELATION COMPUTATION ####################################
    data_summ = data_summ | putils.simple_distance(
            dX, 
            dY, 
            persistence_type=persistence_type,
            q=q, p=p,
            verbose=False
            )
    Wp_XY = data_summ["Wp_XY"]

    nulldX_pathlist = putils._permpaths_from_datapath(os.path.dirname(dX_fpath), nulldir=nulldir, permtype=permtype, verbose=verbose, debug=debug)
    nulldX = [np.loadtxt(fpath.replace("bars_X.txt","dist_mtxs/dX.ldm")) for fpath in nulldX_pathlist]

    nulldY_pathlist = putils._permpaths_from_datapath(os.path.dirname(dY_fpath), nulldir=nulldir, permtype=permtype, verbose=verbose, debug=debug)
    nulldY = [np.loadtxt(fpath.replace("bars_X.txt","dist_mtxs/dX.ldm")) for fpath in nulldY_pathlist]

    permlabels = [putils._parse_pathname(fpath, perm_pathtype=True, debug=False) for fpath in nulldX_pathlist]

    if verbose:
        print(f"X dmtx pulled from: \n{dX_fpath}")
        print(f"Found {len(nulldX)} associated permuted nulls of type \"{permtype}\"")
        print(f"Y dmtx pulled from: \n{dY_fpath}")
        print(f"Found {len(nulldY)} associated permuted nulls of type \"{permtype}\"")

    if debug:
        k=2
        ### debugging code ###
        print(f"Last {k} paths of permuted X nulls: \n{nulldX_pathlist[-k:]}")
        print(f"Last {k} paths of permuted Y nulls: \n{nulldY_pathlist[-k:]}")
        print(f"Restricting to intersection of first {min(len(nulldX), len(nulldY))} permutation sets")
        # assert len(nulldX)==len(nulldY), "Must pair permutations to compute distance distributions"
        ### debugging code ###
    else:
        if verbose:
            print(f"Restricting to intersection of first {min(len(nulldX), len(nulldY))} permutation sets")

    paired_nulldmtx = list(zip(nulldX, nulldY))

    perm_summ = {}
    perm_summ["datatype"] = "Null"
    perm_summ["permtype"] = permtype
    perm_summ["X_type"] = data_summ["X_type"]
    perm_summ["Y_type"] = data_summ["Y_type"]
    pairdist_summ = [data_summ] + [None]*len(paired_nulldmtx)

    if verbose:
        print(f"Computing Wasserstein distances between {len(paired_nulldmtx)} H{homdim} diagrams derived from permutation-matched null data...")

    for i, pair in enumerate(paired_nulldmtx):
        #################################### REPLACE BY DISTANCE CORRELATION COMPUTATION ####################################
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
    
    Wp_XYnull = np.array( [dist["Wp_XY"] for dist in pairdist_summ if dist["datatype"]=="Null"] )
    
#   if len(Wp_XYnull) == 0:
#       empirical_pval = 1
#   else:
#       prop_lower = np.mean(Wp_XY > Wp_XYnull)
#       empirical_pval = max(1/len(Wp_XYnull), 2 * min(prop_lower, 1 - prop_lower))
#
#   pairdist_summ[0]["empirical_pval"] = empirical_pval


    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Datatype of \'X\':", data_summ["X_type"])
        print("Datatype of \'Y\':", data_summ["Y_type"])
        print("Projection cost of sending PD(X) to the empty diagram:", data_summ["PDX_diag"])
        print("Projection cost of sending PD(Y) to the empty diagram:", data_summ["PDY_diag"])
        print(f"Observed Wasserstein distance between X and Y: {Wp_XY}") #, f"p < {empirical_pval}")
        print("Permutation type(s):", ', '.join(list(set([ dist["permtype"] for dist in pairdist_summ if dist["datatype"]=="Null" ]))))
        print(f"Summary of Wasserstein distance from data persistence diagrams to permuted-null persistence modules: \nmu={np.mean(Wp_XYnull)}, sigma={np.std(Wp_XYnull)}")
        print(f"Distribution of Wasserstein distances: \n{np.histogram(Wp_XYnull)}")
        print("\n")



    return pairdist_summ

def comp_dist_stats(dX, dY):
    n = dX.shape[0]

    assert dX.shape == dY.shape, f"Input pairwise distance matrices must have same shape. \ndX={dX.shape}, \ndY={dY.shape}"

    dX_row_means = dX.mean(axis=0, keepdims=True)
    dY_row_means = dY.mean(axis=0, keepdims=True)
    dX_col_means = dX.mean(axis=1, keepdims=True)
    dY_col_means = dY.mean(axis=1, keepdims=True)
    dX_mean = dX.mean()
    dY_mean = dY.mean()

    A = dX - dX_row_means - dX_col_means + dX_mean
    B = dY - dY_row_means - dY_col_means + dY_mean

    S = dX_mean * dY_mean
    dcov = np.sqrt(np.multiply(A, B).mean())
    dvar_x = np.sqrt(np.multiply(A, A).mean())
    dvar_y = np.sqrt(np.multiply(B, B).mean())
    dcor = dcov / np.sqrt(dvar_x * dvar_y)

    test_statistic = n * dcov**2

    return DistDependStat(
        test_statistic=test_statistic,
        distance_correlation=dcor,
        distance_covariance=dcov,
        dvar_x=dvar_x,
        dvar_y=dvar_y,
        S=S,
    )

def _get_outpath(data_summ, outdir=".", permtype="subject"):

    xlabel = data_summ["X_type"]
    ylabel = data_summ["Y_type"]

    outname = f"{xlabel}_vs_{ylabel}_null-{permtype}Perms.csv"
    outpath = os.path.join(outdir, outname).replace("_ztrans","-ztrans")

    return outpath


def _summarize_data(dX_fpath, dY_fpath, debug=True):
    
    xlabels = putils._parse_pathname(os.path.dirname(dX_fpath), perm_pathtype=False, debug=debug)
    ylabels = putils._parse_pathname(os.path.dirname(dY_fpath), perm_pathtype=False, debug=debug)

    data_summ = {}
    data_summ["datatype"] = "Data"
    data_summ["X_type"] = xlabels["modality"] + "_" + xlabels["feature"] + "_" + xlabels["metric"]
    data_summ["Y_type"] = ylabels["modality"] + "_" + ylabels["feature"] + "_" + ylabels["metric"]
    return data_summ

class DistDependStat(NamedTuple):
    test_statistic: float
    distance_correlation: float
    distance_covariance: float
    dvar_x: float
    dvar_y: float
    S: float
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
        "--dX_fpath",
        default=None,
        type=str,
        help="filepath to dmtx of first persistence diagram"
    )
    parser.add_argument(
        "-y",
        "--dY_fpath",
        default=None,
        type=str,
        help="filepath to dmtx of second persistence diagram"
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

    putils._verify_distinct_spaces(args.dX_fpath, args.dY_fpath)

    if args.verbose:
        print(f"\n\nList of datasets pulled from: \n{args.dX_fpath}")
        print(f"List of nullsets pulled from: \n{args.dY_fpath}")
        print(f"Using permutations of type \'{args.permtype}\', from parent directory: \n{args.nulldir}")
        print(f"Computing (p,q)=({args.p},{args.q}) Wasserstein distance between diagrams of generators in H{args.dim}")
        print(f"Persistence values can be defined by either the \'difference\' or \'quotient\' of birth-death pairs: using \'{args.persistence_type}\'\n\n")

    pairdist_summ = get_permset_distcorrs(
            args.dX_fpath, 
            args.dY_fpath,
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

    df_out = pd.DataFrame(pairdist_summ)
    df_out.to_csv(outpath)

    print(f"\n\nSaved to: \n{outpath}")
