import os
import re
import sys
import numpy as np
# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import diagram_distances as dgmD


def_nulldir = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing"

## base computation of Wasserstein distance for diagram generated from null-permuted data
################################################################################################################
def simple_distance(barsX, permbarsX, 
        persistence_type="diff", q=2, p=2, 
        verbose=True, debug=False):

    # compute standard Wasserstein distance between diagrams from X and Y
    Wp_XY = dgmD.weighted_Wasserstein_dist(barsX, permbarsX, wtfn_type=None, q=q, p=p, verbose=verbose)
    PDX_diag = dgmD.weighted_Wasserstein_dist(barsX, None, wtfn_type=None, q=q, p=p, verbose=verbose)
    PDY_diag = dgmD.weighted_Wasserstein_dist(None, permbarsX, wtfn_type=None, q=q, p=p, verbose=verbose)

    results_dict = {
            "Wp_XY": Wp_XY,
            "PDX_diag": PDX_diag, 
            "PDY_diag": PDY_diag, 
            }

    if verbose:
        print("\n")
        print("##########################################    RESULTS SUMMARY    ##########################################")
        print("")
        print("Standard Wasserstein distance between PD(X) and PD(Y):", Wp_XY)
        print("Projection cost of sending PD(X) to the empty diagram:", PDX_diag)
        print("Projection cost of sending PD(Y) to the empty diagram:", PDY_diag)

    return results_dict
################################################################################################################



## cleanup and bookkeeping utility functions
################################################################################################################
# sample datapath: /ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within_ICA/ICA50_pNMs/phom_data_ICA50_pNMs_geodesic_dists/bars_X.txt
# samples for perm_pathtype=True: 
#     - /ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing/within_ICA/ICA50_pNMs/permstrapping/phom_data_ICA50_pNMs_geodesic_dists_subjectPerms_perm_set922_n64620/bars_X.txt
#     - /ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/null_testing/within_ICA/ICA50_Amps/permstrapping/phom_data_ICA50_pNMs_geodesic_dists_featurePerms_seed103/bars_X.txt
def _parse_pathname(datapath, perm_pathtype=False, debug=True):
    labelset={}
    xdirname = os.path.basename(os.path.dirname(datapath))
    xdisttype = xdirname.replace("phom_data_","").split("_dists")[0]
    xdisttype = xdisttype.replace("Psim_ztrans","Psim-ztrans")

    if debug:
        ### debugging code ###
        print(f"xdirname: {xdirname}")
        print(f"xdisttype: {xdisttype}")
        ### debugging code ###

    modality, feature, metric = xdisttype.split("_")
    labelset["modality"] = modality
    labelset["feature"] = feature
    labelset["metric"] = metric
    labelset["datatype"]="Data"

    if perm_pathtype:
        permtype = xdirname.split("_dists")[1]
        permlabel = permtype.split("Perms_")[1]
        if "subject" in permtype:
            permtype = "subject"
        elif "feature" in permtype:
            permtype = "feature"
        else:
            raise ValueError(f"unrecognized permutation type: {permtype}")

        labelset["datatype"] = "Null"
        labelset["permtype"] = permtype
        labelset["permlabel"] = permlabel
    return labelset


def _permpaths_from_datapath(datapath, nulldir=def_nulldir, permtype="subject", permno="*", debug=False, verbose=False):
    import re
    import glob
    labelset = _parse_pathname(datapath, perm_pathtype=False, debug=False)

    modality = labelset["modality"]
    modality_nonum = re.sub(r'[0-9]+','', modality)
    feature = labelset["feature"]
    metric = labelset["metric"].replace('Psim-ztrans','Psim_ztrans')

    if permtype=="subject":
        permlabel = f"perm_set{permno}_n64620"
    elif permtype=="feature":
        permlabel = f"seed{permno}"
    else:
        raise ValueError(f"Unrecognized permutation type \"{permtype}\"")

    # following procedure assumes naming convention follows those used by scripts in `src_bash`
    gen_permpath = os.path.join(
            nulldir, 
            f"within_{modality_nonum}",
            f"{modality}_{feature}",
            "permstrapping",
            f"phom_data_{modality}_{feature}_{metric}_dists_{permtype}Perms_{permlabel}",
            "bars_X.txt")

    permpaths = glob.glob(gen_permpath)
    permpaths.sort()

    if verbose:
        print(f"parent null directory: \n{nulldir}")
        print(f"matching files with general form: \n{gen_permpath}")
        print(f"Found {len(permpaths)} matches.")
    return permpaths

def _check_and_clean_barspath(inpath, verbose=True):
    if inpath.endswith("bars_X.txt"):
        barpath = inpath
        if verbose:
            print(f"Path appears to already point to a file of persistence bars: \n{inpath}\n")
    elif inpath.endswith(".txt") and "dist" in os.path.basename(inpath):
        if verbose:
            print(f"Converting distance path to persistence bar path : \n{inpath}")
        barname = "phom_data_" + os.path.basename(inpath.replace(".txt",""))
        barpath = os.path.join( os.path.dirname(inpath), barname, "bars_X.txt")
    elif os.path.isdir(inpath):
        if verbose:
            print(f"Input path is directory (assumed to contain persistence bar file): \n{inpath}")
        barpath = os.path.join(inpath, "bars_X.txt")
    else:
        if verbose:
            print(f"Input path not of recognized type (passing forward unchanged): \n{ipath}")
        barpath = inpath

    return barpath
################################################################################################################


## debugging code
################################################################################################################
### debugging code ###
def _debug_bars_list(bars_flist, bars_list, name="X-hat_i"):
    Xlen=[len(bars) for bars in bars_list]
    print(f"Pulled {name} bars from {len(bars_flist)} files.")
    print(f"\nNumber of bars in {name} is distributed as:")
    print(np.histogram(Xlen))
    print("")
    if 0 in Xlen:
        print(f"full list of bar numbers: {Xlen}")
        print(f"Empty {name} diagrams come from the following files:")
        for i,num in enumerate(Xlen):
            if num==0:
                print(bars_flist[i])
### debugging code ###
################################################################################################################



## Load-in checker: skips computations for what should be trivial distances
################################################################################################################
def _verify_distinct_spaces(pathX, pathY):
    if pathX==pathY:
        print("Trivial case! The following path were given as seperate identical inputs for distinct spaces:")
        print(pathX)
        print("\n\nExiting.")
        exit()
################################################################################################################
