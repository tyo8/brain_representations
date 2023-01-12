import os
import sys
import csv
import numpy as np

sys.path.append('/scratch/tyoeasley/brain_representations/src_py')
import phom_figs as pf
import HCP_utils as hutils
import comp_phom as cph


def naive_phom(distlist_fname, out_dir, dist_ord=1):
    dist_fname_list = hutils.load_namelist(distlist_fname)
    namelist = [os.path.splitext(os.path.basename(fname))[0] for fname in dist_fname_list]

    dist_prefix = 'p' + str(dist_ord) + 'dist_'
    
    PD_fname = os.path.join(out_dir, dist_prefix + 'persistence_dgms.npy')
    return PD_fname, namelist
  


def benchmark_W2dists(namelist, PD_array=[], PD_fname='', outdir='', dist_ord=1, make_figs=False):
    
    dist_prefix = 'p' + str(dist_ord) + 'dists_'

    print('Running with make_figs=' + str(make_figs))

    if not PD_array:
        PD_array = np.load(PD_fname)

    print('PD_array has shape: ' + str(PD_array.shape))

    if not outdir:
        outdir = os.path.dirname(PD_fname)

    import datetime 
    start = datetime.datetime.now()
    print("Exporting Wasserstein diagram distances...")
    W2_dists = cph.comp_phom_dists(phom_dgms=PD_array)   # Wasserstein distance matrix (pairwise)
    np.save(os.path.join(out_dir, dist_prefix + 'phom_dists.npy'), W2_dists)
    lapse = datetime.datetime.now() - start
    print('Done. (' + str(lapse) + ' elapsed)')
    print('')

    if make_figs:
        start = datetime.datetime.now()
        print("Visualizing pairwise W2 distance matrices...")
        pf.export_dists(out_dir, phom_dists=W2_dists,
                fname_prefix= 'W2_dists_' + dist_prefix + 'H')
        lapse = datetime.datetime.now() - start
        print('Done. (' + str(lapse) + ' elapsed)')
        print('')
    


def distlist_from_simlist(simlist_fname, p = 1):
    distlist_fname = simlist_fname.replace('sim','dist')
    return distlist_fname


if __name__=="__main__":
    simlist_fname=sys.argv[1]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(simlist_fname)), 'phoms')

    if not os.path.isdir(out_dir):
        print("WARNING: created directory " + out_dir)
        os.makedirs(out_dir)

    pset = [2]

    PD_fnames = [None]*len(pset)
    distnames = [None]*len(pset)

    for i in range(len(pset)):
        distlist_fname = distlist_from_simlist(simlist_fname, p = pset[i])
        PD_fnames[i],distnames[i] = naive_phom(distlist_fname, out_dir, dist_ord = pset[i])
        print('PD_fnames:')
        print(PD_fnames[i])
        print('distnames:')
        print(distnames[i])


    for i in range(len(pset)):
        benchmark_W2dists(distnames[i], PD_fname=PD_fnames[i], dist_ord = pset[i], make_figs=True)
