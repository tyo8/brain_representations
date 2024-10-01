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
        perm_path = perm_set
        n_feats = data.shape[1]
        perm_set = np.loadtxt(perm_path, max_rows=n_feats).astype(int)
        if perm_set.shape[0] < n_feats and perm_type=="subject":
            perm_set = get_oversize_perm_set(n_feats, perm_path=perm_path)
        shape_err=f"Permutation set for \'subject\'-type permutations should have shape of transpose of data:\nperm shape: {perm_set.shape} \ndata shape: {data.shape}"
        assert np.all(data.shape==perm_set.shape[::-1]), shape_err
    else:
        assert isinstance(perm_set, np.ndarray), "Unrecognized input type for permutation set"

    if perm_type=="subject":
        permdata = _permute_subj(data, perm_set, check=check)
    elif perm_type=="feature":
        permdata = _permute_feat(data, perm_set, rng_seed, check=check)
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
        Warning("No permutation set given: assuming data is freely exchangeable along intended ('feature'?) axis.")
        rng = np.random.default_rng(rng_seed)
        permdata = np.asarray([rng.permutation(data_i) for data_i in data])
    elif isinstance(perm_set, np.nadarray):
        permdata = np.array([data[i][perm] for i,perm in enumerate(perm_set)])

    if check:
        print(f"Subject distribution peresrved: {np.all([np.sort(permdata[i])==np.sort(data[i]) for i in range(data.shape[1])])}")
    return permdata
###################################################################################################


###################################################################################################
def generate_subj_perms(EB_fname, outdir, perms, sets, overwrite=False):
    # load exchangeability blocks from file
    EBs = np.loadtxt(EB_fname, delimiter=",").astype(int)

    ### debugging code ###
    print(f"\nRequested {sets} sets of {perms} permutations.\n")
    ### debugging code ###

    # compute and save out permutation sets
    for i in range(sets):
        if not os.path.isfile(f"perm_set{i}_n{perms}.csv") or overwrite:
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
            np.savetxt(outpath, perm_set.astype(int), fmt='%i')
###################################################################################################


###################################################################################################
def get_oversize_perm_set(n_feats, perm_path=None, debug=True):
    if perm_path is not None:
        perm_dir = os.path.dirname(perm_path)
        set_num = int(os.path.basename(perm_path).split("perm_set")[1].split('_')[0])
        n_perms_per_set = int(os.path.basename(perm_path).split("_n")[1].split('.')[0])
    else:
        perm_dir = "/scratch/tyoeasley/brain_representations/phom_analysis/null_testing/permutations/subject_perms"
        set_num = 0
        n_perms_per_set = 64620
    
    rng = np.random.default_rng(set_num)
    n_sets = int(np.ceil(n_feats/n_perms_per_set))
    perm_sets = [None]*n_sets

    all_perms_list = [os.path.join(perm_dir,f) for f in os.listdir(perm_dir) if "perm_set" in f]
    all_perms_list.sort()

    perm_paths = rng.permutation(all_perms_list)[:n_sets]

    ### debugging code ###
    if debug:
        print(f"\nInitially attempted pull from permutation set at path: \n{perm_path}")
        print(f"Pulling {n_sets} perm sets, each of length {n_perms_per_set}, to define perms for {n_feats} within-data subject permutations:\n")
        for i in perm_paths:
            print(i)
        print("")
    ### debugging code ###

    for idx, fpath in enumerate(perm_paths):
        permset_i = np.loadtxt(fpath).astype(int)
        perm_sets[idx] = rng.permutation(permset_i, axis=0)     # assumes permset_i has shape (n_perms_per_set, n_subjects)
        
    all_perm_set = np.concatenate(perm_sets, axis=0)[:n_feats]

    return all_perm_set
###################################################################################################

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
    parser.add_argument(
        "-w", "--overwrite", default=False, action="store_true", help="if flag given, then overwrite existing permutation sets with same path"
    )
    args = parser.parse_args()

    if args.generate:
        generate_subj_perms(
                args.EB_fname,
                args.output_dir,
                args.perms,
                args.perm_sets,
                overwrite = args.overwrite
                )
