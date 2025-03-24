import os
import sys
import argparse
import numpy as np
import pandas as pd
import permtest_utils as putils

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD

## Primary functions
################################################################################################################
def get_permset_dists(datapath, permbarsX_listpath,
        persistence_type="diff", homdim=1, q=2, p=2,
        match_perms=True, verbose=True, debug=True):
    
    barsX = dgmD._get_bars(datapath, homdim=homdim)
    if match_perms:
        permpath_list = putils._permpaths_from_datapath(datapath, permno="*", verbose=verbose, debug=debug)
    else:
        with open(permbarsX_listpath, 'r') as fin:
            permpath_list = [ putils._check_and_clean_barspath(ppath, verbose=False) for ppath in fin.read().split() ]
            if debug:
                ### debugging code ###
                print(f"Datapath to bars for permuted dataset (first 11 entries): \n{permpath_list[0:10]}")
                ### debugging code ###

    permdists_out = [None]*len(permpath_list)


    for i, permpath in enumerate(permpath_list):
        perm_labels = putils._parse_pathname(permpath, perm_pathtype=True, debug=debug)


        permbarsX = dgmD._get_bars(permpath, homdim=homdim)
        dist_summ = putils.simple_distance(
                barsX, 
                permbarsX,
                persistence_type=persistence_type,
                q=q, p=p,
                verbose=False
                )
        # do something to collate dist_summ
        permdists_out[i] = perm_labels | dist_summ

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Permutation type(s):", set([ dist["permtype"] for dist in permdists_out ]))
        Wp_X_nullX = np.array( [dist["Wp_XY"] for dist in permdists_out] )
        print(f"Distribution of Wasserstein distance from data persistence diagrams to permuted-null persistence modules: \nmu={np.mean(Wp_X_nullX)}, sigma={np.std(Wp_X_nullX)}")
        PDX_diag = np.mean(np.array( [dist["PDX_diag"] for dist in permdists_out] ))
        print(f"Wasserstein distance from data persistence module to the diagonal diagram: \n{PDX_diag}")
        PDnullX_diag = np.array( [dist["PDY_diag"] for dist in permdists_out] )
        print(f"Wasserstein distance from permuted-null persistence modules to the diagonal diagram: \nmu={np.mean(PDnullX_diag)}, sigma={np.std(PDnullX_diag)}")
        print("\n")



    return permdists_out

def _write_out(outdir, df_full, verbose=True):

    keynames = ["modality", "feature", "permtype", "metric"]


    for mod in list(set(df_full["modality"])):
        df = df_full[ df_full["modality"] == mod ]

        for feat in list(set(df["feature"])):
            df = df[ df["feature"] == feat]

            for perm in list(set(df["permtype"])):
                df = df[ df["permtype"] == perm]

                for dist in list(set(df["metric"])):
                    df = df[ df["metric"] == dist]
                    outname = f"data_vs_{perm}null_{mod}_{feat}_{dist}.csv"
                    outpath = os.path.join(outdir, outname)
                    df.to_csv(outpath)
                    print(f"Results written to {outpath}")


    return outpath
################################################################################################################



################################################################################################################
# parses input, saves output
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Show distributions of outputs from topological bootstrap"
    )
    parser.add_argument(
        "-x",
        "--datapath",
        default=None,
        type=str,
        help="filepath to bars of first persistence diagram"
    )
    parser.add_argument(
        "-y",
        "--nullpath_lists",
        default=None,
        type=str,
        help="filepath to bars of second persistence diagram"
    )
    parser.add_argument(
        "-o", "--outdir",
        default=None, 
        type=str, 
        help="output directory"
    )
    parser.add_argument(
        "-d", "--dim", 
        default=1, 
        type=int, 
        help="homology dimension"
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
    parser.add_argument(
        "-m", "--match_perms", 
        default=False, 
        action="store_true",
        help="toggle: enforce match between modality, feature, and distance metric of data vs. null"
    )
    args = parser.parse_args()

    if args.verbose:
        print(f"List of datasets pulled from: \n{args.datapath}")
        print(f"List of nullsets pulled from: \n{args.nullpath_lists}")

    null_dists = get_permset_dists(
            args.datapath, 
            args.nullpath_lists,
            persistence_type=args.persistence_type, 
            homdim=args.dim, 
            q=args.q, 
            p=args.p,
            match_perms=args.match_perms,
            verbose=args.verbose 
            )
    
    df_full = pd.DataFrame(null_dists)
    outpath = _write_out(args.outdir, df_full, verbose=args.verbose)

    print(f"\n\nSaved to: \n{outpath}")
