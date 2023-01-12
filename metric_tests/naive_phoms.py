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
    dist_mtx_list = [np.genfromtxt(fname) for fname in dist_fname_list]

    dist_prefix = 'p' + str(dist_ord) + 'dist_'
    ## debug code ##
    # for i in dist_fname_list:
    #     print(i)
    #     np.genfromtxt(i, delimiter=",")
    #     print([i.shape for i in dist_mtx_list])
    ## debug code ##

    PD_array = cph.compute_phom(dist_mtx_list, hom_dims=(0,1,2), benchmark=True)              # persistence diagrams
    PD_fname = os.path.join(out_dir, dist_prefix + 'persistence_dgms.npy')
    np.save(PD_fname, PD_array)
    return PD_fname, namelist



def make_dgms(namelist, PD_array=[], PD_fname='', outdir=''):
    if not PD_array:
        PD_array = np.load(PD_fname)

    if not outdir:
        outdir = os.path.dirname(PD_fname)

    PD_fnamelist = [os.path.join(out_dir, i + '.html') for i in namelist]
    pf.plot_PHdgms(PD_array, PD_fnamelist, namelist)


    
def make_bcurves(namelist, PD_array=[], PD_fname='', outdir='', dist_prefix='p1dist_', make_figs=False):
    if not PD_array:
        PD_array = np.load(PD_fname)

    if not outdir:
        outdir = os.path.dirname(PD_fname)

    BC_array = cph.comp_Betti_curves(phom_dgms=PD_array, write_mode=False, plot=False)    # Betti curves
    np.save(os.path.join(out_dir, dist_prefix + 'Betti_curves.npy'), BC_array)
    if make_figs:
        pf.export_bcurves(out_dir, betti_curveset=BC_array, labelset=namelist, linestyle='.:',
                fname_prefix='Bcurve_' + dist_prefix + 'H', title_suffix='varying distance metrics')
    


def benchmark_exports(namelist, PD_array=[], PD_fname='', outdir='', dist_ord=1, make_figs=False):
    import datetime

    print('Running with make_figs=' + str(make_figs))

    dist_prefix = 'p' + str(dist_ord) + 'dists_'

    if make_figs:
        start = datetime.datetime.now()
        print("Exporting persistence diagrams...")
        make_dgms(namelist, PD_array=PD_array, PD_fname=PD_fname, outdir=outdir)
        lapse = datetime.datetime.now() - start
        print('Done. (' + str(lapse) + ' elapsed)')
   
    start = datetime.datetime.now()
    print("Exporting Betti curves...")
    make_bcurves(namelist, PD_array=PD_array, PD_fname=PD_fname, outdir=outdir, dist_prefix=dist_prefix, make_figs=make_figs)
    lapse = datetime.datetime.now() - start
    print('Done. (' + str(lapse) + ' elapsed)')
    print('')



def distlist_from_simlist(simlist_fname, p = 1):
    distlist_fname = simlist_fname.replace('sim','dist')
    
    sim_fname_list = hutils.load_namelist(simlist_fname)

    ## debug code ##
    # for i in sim_fname_list:
    #     print(i)
    #     np.genfromtxt(i, delimiter=",")
    ## debug code ##

    sim_mtxs = [np.genfromtxt(i) for i in sim_fname_list]
    dist_mtxs = [hutils.p_dist(i, order=p) for i in sim_mtxs]

    ## debug code ##
    print('distance matrix shape ' + str(dist_mtxs[0].shape))
    print('distance matrix nanflag: ' + str(np.any(np.isnan(dist_mtxs[0]))))
    ## debug code ##

    dist_fname_list = [fname.replace('sims.txt','p'+ str(p) +'dists.txt') if "sims" in fname else fname for fname in sim_fname_list]

    for k in range(len(dist_mtxs)):
        np.savetxt(dist_fname_list[k], dist_mtxs[k])

    with open(distlist_fname, 'w') as fout:
        for fname in dist_fname_list:
            fout.write('%s\n' % fname)

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
        # PD_fnames[i],distnames[i] = naive_phom(distlist_fname, out_dir, dist_ord = pset[i])


    for i in range(len(pset)):
        # benchmark_exports(distnames[i], PD_fname=PD_fnames[i], dist_ord = pset[i], make_figs=True)
