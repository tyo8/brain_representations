import os
import sys
import skpalm
import argparse
import numpy as np

## DEFINE PARENT "PERMUTE" FUNCTION THAT INCLUDES LOGIC FOR BOTH "subject" AND "feature" TYPE PERMUTATIONS
###################################################################################################
def permute(data, perm_type="subject", perm_set=None, rng_seed=0, check=False):
    if perm_set is None:
        perm_set = None
    elif isinstance(perm_set, str):
        n_feats = data.shape[1]
        perm_set = np.loadtxt(perm_set, max_rows=n_feats).astype(int)
        shape_err=f"Permutation set for \'subject\'-type permutations should have shape of transpose of data:\nperm shape: {perm_set.shape} \ndata shape: {data.shape}"
        assert np.all(data.shape==perm_set.shape[::-1]), shape_err
    else:
        assert isinstance(perm_set, np.ndarray), "Unrecognized input type for permutation set"

    if perm_type=="subject":
        permdata = _permute_subj(data, perm_set, check=check)
    elif perm_type=="feature":
        permdata = _permute_subj(data, perm_set, rng_seed, check=check)
    else:
        raise ValueError(f"Unrecognized permutation type \"{perm_type}\".")

    return permdata

# assumes that 'data' has shape (subjects)x(features) and 'perm_set' has shape (features)x(subjects)
def _permute_subj(data, perm_set, check=True):
    if perm_set is None:
        raise ValueError("A structure-respecting permutation set must be given for \"subject\"-type permutations in the HCP: value \'{perm_set}\' was given instead.")
    data=data.T
    ### debugging code ###
    print(f"maximum and minimum indices in permutation set: {( perm_set.min(), perm_set.max() )}")
    ### debugging code ###
    permdata = np.array([data[i][perm] for i,perm in enumerate(perm_set)]).T

    ### debugging code ###
#   for i,perm in enumerate(perm_set):
#       print(f"permutation set (shape {perm.shape}): \n{perm}")
#       print(f"unpermuted data (shape {data[i].shape}): \n{data[i]}")
#       print(f"permuted data (shape {data[i][perm].shape})): \n{data[i][perm]}")
    ### debugging code ###

    if check:
        data=data.T
        print(f"Feature distribution preserved: {np.all([np.sort(permdata[:,i])==np.sort(data[:,i]) for i in range(data.shape[1])])}")
    return permdata


def _permute_feat(data, perm_set, rng_seed, check=True):
    if perm_set is None:
        Warning("No permutation set given: assuming data is freely exchangeable.")
        rng = np.random.default_rng(rng_seed)
        permdata = np.asarray([rng.permutation(data_i) for data_i in data])
    elif isinstance(perm_set, np.nadarray):
        permdata = np.array([data[i][perm] for i,perm in enumerate(perm_set)])

    if check:
        print(f"Subject distribution peresrved: {np.all([np.sort(permdata[i])==np.sort(data[i]) for i in range(data.shape[1])])}")
    return permdata
###################################################################################################


###################################################################################################
def generate_subj_perms(EB_fname, outdir, perms, sets):
    # load exchangeability blocks from file
    EBs = np.loadtxt(EB_fname, delimiter=",").astype(int)

    ### debugging code ###
    print(f"\nRequested {sets} sets of {perms} permutations.\n")
    ### debugging code ###

    # compute and save out permutation sets
    for i in range(sets):
        # call scikit-palm to compute permutation set
        perms_out = skpalm.permutations.quickperms(
                exchangeability_blocks=EBs, 
                perms=perms, 
                ignore_repeat_perms=True,
                verbose=True
                )

        perm_set = perms_out[0].T

        # force compliance with 0-indexing convention
        if perm_set.min() == 1:
            perm_set -= 1

        outpath = os.path.join(outdir, f"perm_set{i}_n{perms}.csv")
        np.savetxt(outpath, perm_set)
###################################################################################################

## ADD PRE-PARSER THAT LOOKS FOR "G" (for "generate") FLAG MAYBE??

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Generate family structure-respecting permutations of HCP data"
    )
    parser.add_argument(
        "-G", "--generate", default=False, action="store_true", help="if flag given, then generate and save permutation sets "
    )
    parser.add_argument(
        "-i", "--EB_fname", type=str, help="input filepath to saved exchangeability blocks"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="output directory for permutation sets"
    )
    parser.add_argument(
        "-n", "--perms", default=10**5, type=int, help="number of requested permutations in a given permuation set"
    )
    parser.add_argument(
        "-N", "--perm_sets", default=100, type=int, help="number of sets of permutations requested"
    )
    args = parser.parse_args()

    if args.generate:
        generate_subj_perms(
                args.EB_fname,
                args.output_dir,
                args.perms,
                args.perm_sets
                )
