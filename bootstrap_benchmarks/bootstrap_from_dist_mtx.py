import os
import sys
import dill
import datetime
import numpy as np

sys.path.append('/scratch/tyoeasley/brain_representations/src_py/interval-matching-precomp_metric/match/utils_PH')
import mult_matches as mm

def main(dist_mtx_fname, output_dir, n_samps=250, bootstrap_prop = 0.8, 
        par_flag=True, n_workers=12, verbose_out=True): 
    dist_mtx = np.genfromtxt(dist_mtx_fname)
    nb_X = dist_mtx.shape[0]

    bootstrap_n = round(nb_X * bootstrap_prop)
    idx_arr = np.random.choice(nb_X, (bootstrap_n, n_samps))

    # gives identifier to saveouts; apply to all except phom_X, which should be invariant under all runs
    timestamp = datetime.datetime.now().strftime('%Y%h%d_%H_%M_%S')
    print('Running data marked with start-time stamp: ' + timestamp)

    if par_flag:
        subidx_list = np.array_split(idx_arr, n_workers, axis=1)
        inp_list = [[dist_mtx_fname, subidx_arr, verbose_out] for subidx_arr in subidx_list]

        import pathos.multiprocessing as mp
        with mp.ProcessingPool(n_workers) as P:
            w_matched_X_Xi, w_affinity_X_Xi, w_phom_X, w_phom_Xi = zip(
                    *P.map(call_bootstrap_matching, inp_list)
                    )
            all_matched_X_Xi = _unpack_worker_returns(w_matched_X_Xi)
            all_affinity_X_Xi = _unpack_worker_returns(w_affinity_X_Xi)   # list of affinity values
            phom_X = w_phom_X[0]                                            # in format [bars, reps, tight_reps]
            all_phom_Xi = _unpack_worker_returns(w_phom_Xi)               # in format {list of [bars, reps, tight_reps] per i}
    else:
        all_matched_X_Xi, all_affinity_X_Xi, phom_X, all_phom_Xi = call_bootstrap_matching([dist_mtx_fname, idx_arr, verbose_out])

    # NOTE: some kind of saveout stuff needs to happen here
    match_savename = 'all_matched_' + timestamp + '.gens'
    with open(os.path.join(output_dir, match_savename), 'wb') as fout:
        dill.dump(all_matched_X_Xi, fout)

    affinity_savename = 'all_affinity_' + timestamp + '.vals'
    with open(os.path.join(output_dir, affinity_savename), 'wb') as fout:
        dill.dump(all_affinity_X_Xi, fout)

    Xphom_savename = 'X.phom'
    with open(os.path.join(output_dir, Xphom_savename), 'wb') as fout:
        dill.dump(phom_X, fout)

    Xiphom_savename = 'all_Xi_' + timestamp + '.phom'
    with open(os.path.join(output_dir, Xiphom_savename), 'wb') as fout:
        dill.dump(all_phom_Xi, fout)

    return None 


def call_bootstrap_matching(inp_list):
    dist_mtx_fname = inp_list[0]
    subidx_arr = inp_list[1]
    verbose_out = inp_list[2]
    subidx_list = [subidx_arr[:,i] for i in range(subidx_arr.shape[1])]

    dX = np.genfromtxt(dist_mtx_fname)

    ### debug code ###
    print('Loading data from ' + dist_mtx_fname)
    print('Distance matrix has shape ' + str(dX.shape))
    # print('subset indices have shapes ' + str([i.shape for i in subidx_list]))
    # print('Sample subset index: ' + str(subidx_list[0]))
    ### debug code ###
    
    list_matched_X_Xi, list_affinity_X_Xi, list_phom_X, list_phom_Xi = mm.bootstrap_matching(
            dX=dX, subidx_list=subidx_list, verbose_out=verbose_out
            )
        
    return list_matched_X_Xi, list_affinity_X_Xi, list_phom_X, list_phom_Xi


def _unpack_worker_returns(worker_return_list):
    unpacked_list = [list_item for worker_return in worker_return_list for list_item in worker_return]
    return unpacked_list


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute diffusion embedding of data for every subject in list')
    parser.add_argument('dist_mtx_fname',
            help='Path to .csv or .txt file containing the entries of the subject-by-subject distance matrix') 
    parser.add_argument('-o','--out',
            dest='outdir', type=str,
            default = '/scratch/tyoeasley/brain_representations/bootstrap_benchmarks/tst/phoms',
            help='string containing output directory')
    parser.add_argument('-s --samples',
            dest='n_samps', default=250, type=int,
            help='number of bootstraps to use (default=250)')
    parser.add_argument('-p', '--bstrap_prop',
            dest='bootstrap_prop', default=0.80, type=float,
            help='proportion of data to use in each bootstrap (default=0.9)')
    parser.add_argument('--nopar',
            dest='par_flag', default=True, action='store_const', const=False,
            help='Logical flag: to distribute or not to distribute?') 
    parser.add_argument('-n', '--n_workers',
            dest='n_workers', default=12, type=int,
            help='Number of parallel workers assigned to computation')
    parser.add_argument('-v','--verbose',
            dest='verbose', default=False, action='store_const', const=True,
            help='Logical flag: if passed, give verbose output')
    args = parser.parse_args()
    
    main(
            args.dist_mtx_fname, args.outdir, n_samps=args.n_samps, bootstrap_prop = args.bootstrap_prop, 
            par_flag=args.par_flag, n_workers=args.n_workers, verbose_out=args.verbose
            )
