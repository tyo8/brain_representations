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
  


def benchmark_bcurves(namelist, PD_array=[], PD_fname='', outdir='', dist_ord=1, make_figs=False):
    
    dist_prefix = 'p' + str(dist_ord) + 'dists_'

    print('Running with make_figs=' + str(make_figs))

    if not PD_array:
        PD_array = np.load(PD_fname)

    print('PD_array has shape: ' + str(PD_array.shape))

    if not outdir:
        outdir = os.path.dirname(PD_fname)

    import datetime 
    start = datetime.datetime.now()
    print("Computing Betti curves...")
    BC_array = cph.comp_Betti_curves(phom_dgms=PD_array, write_mode=False, plot=False)    # Betti curves
    np.save(os.path.join(out_dir, dist_prefix + 'Betti_curves.npy'), BC_array)
    lapse = datetime.datetime.now() - start
    print('Done. (' + str(lapse) + ' elapsed)')
    print('')

    if make_figs:
        start = datetime.datetime.now()
        print("Plotting Betti curves...")
        pf.export_bcurves(out_dir, betti_curveset=BC_array, labelset=namelist, linestyle='.:',
                fname_prefix='Bcurve_' + dist_prefix + 'H', title_suffix='varying distance metrics')
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
        benchmark_bcurves(distnames[i], PD_fname=PD_fnames[i], dist_ord = pset[i], make_figs=True)
