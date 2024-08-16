import gc
import os
import sys
import csv
import numpy as np
import phom_figs as pf
import gtda.homology as hml
from gtda import diagrams as dgms


def main(argvals):
    dist_list_fname = argvals[1]
    phom_dgms_fname = argvals[2]
    phom_dist_fname = argvals[3]
    betti_curves_fname = argvals[4]
    perst_imgs_fname = argvals[5]
    comp_persdiag(dist_list_fname,phom_dgms_fname)
    comp_phom_dists(phom_dgms_fname,fname_out=phom_dist_fname)
    comp_Betti_curves(fname_in=phom_dgms_fname, fname_out=betti_curves_fname)
    comp_perst_imgs(phom_dgms_fname,fname_out=perst_imgs_fname)


## receives name of csv file storing a list of names of distance matrix files,
## then computes and saves a set of persistence diagrams
def comp_persdiag(dist_flist_name,fname_out, normalize = True):
    with open(dist_flist_name,newline='') as fin:
        dist_flists = list(csv.reader(fin))
        dist_flist = list(map(''.join,dist_flists))

    dist_mtx_input = [np.loadtxt(i, delimiter=",") for i in dist_flist]
    persdiag = compute_phom(dist_mtx_input, normalize=normalize)

    np.save(fname_out,persdiag)
    print("Persistence diagrams computed from "+dist_flist_name+" and saved to "+fname_out)

    pf.export_PHdgms(dist_flist_name,fname_out)


def compute_phom(dist_mtx_input, hom_dims=(0, 1, 2, 3), normalize=True, benchmark=False):
    # dimensions of homology groups
    # hom_dims = (0, 1)
    # hom_dims = (0, 1, 2)
    # hom_dims = (0, 1, 2, 3)       # the cluster memory can't (usually) handle hom_dim>3 for a 1003-subj Gram matrix, even with edge collapse

    if benchmark:
        import datetime
        start = datetime.datetime.now()
        print("Computing persistence diagrams...")


    dist_mtx_input = clean_dist_input(dist_mtx_input, normalize=normalize)

    # give simplicial complex structure
    VR = hml.VietorisRipsPersistence(
    	metric = "precomputed",
        homology_dimensions = hom_dims,
        collapse_edges = True
    )


    # compute persistence diagram (output as [#dist_mtxs] x N x [len(hom_dims)] array)
    print("Computing persistence diagrams for dimensions " + str(hom_dims) + "...")
    persdiag = VR.fit_transform(dist_mtx_input)

    print('Found a maximum of ' + str(persdiag.shape[1]) + ' generators (in all computed dimensions) over ' + str(persdiag.shape[0]) + ' diagrams.')

    if benchmark:
        lapse = datetime.datetime.now() - start
        print('Done. (' + str(lapse) + ' elapsed)')
    else:
        print("Done.")

    print('')

    del VR
    gc.collect()

    return persdiag


def clean_dist_input(dist_mtx_input, normalize=True):    
    if isinstance(dist_mtx_input,np.ndarray):
        if dist_mtx_input.ndim <= 2:
            if normalize:
                dist_mtx_input = dist_mtx_input/np.amax(dist_mtx_input) 

            dist_mtx_input=dist_mtx_input[None,:,:]
        else:
            if normalize:
                for j in range(dist_mtx_input.shape[2]):
                    dist_mtx_input[:,:,j]=dist_mtx_input[:,:,j]/np.amax(dist_mtx_input[:,:,j])
            
        print("dist_mtx_input has dimension " + str(dist_mtx_input.shape))
    else:
        if normalize:
            dist_mtx_input = [i/np.amax(i) for i in dist_mtx_input]
        n_samps = len(dist_mtx_input)
        print("dist_mtx_input has dimension " + str(n_samps) + "x" + str(dist_mtx_input[0].shape))

    return dist_mtx_input 



def comp_phom_dists(fname_in='', phom_dgms=np.asarray([]), 
        fname_out="phoms_dists.npy", write_mode=False,
        n_jobs=1, delta=0.0001, metric="wasserstein"):

    PW = dgms.PairwiseDistance(metric=metric,
            metric_params={'delta': delta},
            order=None, n_jobs=n_jobs)
    
    if not phom_dgms.any():
        phom_dgms= np.load(fname_in)
    phom_dist_mtx = PW.fit_transform(phom_dgms)
    
    if write_mode:
        np.save(fname_out,phom_dist_mtx)
        fig_dir = os.path.dirname(fname_out)+"/figures"
        pf.export_dists(fname_out,fig_dir)
    return phom_dist_mtx


def comp_Betti_curves(fname_in='', phom_dgms=np.asarray([]), 
        fig_dir="figures", fname_out="Betti_curves.npy",
        n_bins=250, n_jobs=1, write_mode=True, plot=True):

    if not phom_dgms.any():
        phom_dgms= np.load(fname_in)

    BC = dgms.BettiCurve(n_bins=n_bins, n_jobs=n_jobs)

    betti_curves = BC.fit_transform(phom_dgms)  # computes Betti curves from phom data
    filt_pars    = _get_filt_pars(BC)            # filtration parameters corresponding to phom data

    betti_curveset = np.append(betti_curves,filt_pars,axis = 0)

    if write_mode:
        np.save(fname_out,betti_curveset)

    if plot:
        fig_dir = os.path.join(os.path.dirname(fname_out), fig_dir)
        pf.export_bcurves(fig_dir, betti_curveset_name=fname_out)
    return betti_curveset

def _get_filt_pars(fit_transform_obj):
    samplings_dict = fit_transform_obj.samplings_
    ndims  = len(samplings_dict)
    nsamps = len(samplings_dict[0])
    samplings = np.zeros((ndims,nsamps))
    for i in range(ndims):
        samplings[i,:] = samplings_dict[i]

    samplings = np.expand_dims(samplings,0)
    return samplings
    

def comp_perst_imgs(fname_in='', phom_dgms=np.asarray([]),
        fname_out="perst_imgs.npy", write_mode=False,
        sigma=0.05, n_bins=400, n_jobs=-1):

    if not phom_dgms.any():
        phom_dgms= np.load(fname_in)

    PI = dgms.PersistenceImage(
            sigma=sigma,n_bins=n_bins,
            n_jobs=n_jobs)

    phom_dgms  = np.load(fname_in)
    perst_imgs = PI.fit_transform(phom_dgms)

    if write_mode:
        np.save(fname_out,perst_imgs)
        fig_dir = os.path.dirname(fname_out)+"/figures_tst/PHimgs"
        pf.export_imgs(fname_out,fig_dir)
    return perst_imgs


if __name__=="__main__":
    main(sys.argv)
