"""
Created on Fri Oct  5 15:48:51 2018
"""
# 

import os
import sys
import numpy as np
import compute_subj_gradients as cgrad


def group_diffusion_maps(gdconn_fpath, grad_fpath, options):
    gdconn = np.single(np.load(gdconn_fpath))
    gdconn = trim_dconn(gdconn)
    group_aff = cgrad.dconn_to_affinity(gdconn)
    emb,res = cgrad.comp_diffusion_embedding(group_aff, 
            alpha=options.alpha, 
            n_components=options.n_comp)
    grad_fpath_emb = cgrad.export_gradients(grad_fpath, emb, res)
    cgrad.export_cifti(grad_fpath_emb, emb.T, headers_list=None)

def trim_dconn(dconn, sub_cort_idx_start=59412):
    # connectivity matrix should be square
    N = dconn.shape[0]
    dconn = np.delete(np.delete(dconn, range(sub_cort_idx_start,N), axis=0), 
            range(sub_cort_idx_start,N), axis=1)

    # medmask = cgrad.get_medial_mask(medial_mask_fpath)
    #DEPRECATED: delete connectivity corresponding to medial wall
    # dconn = np.delete(np.delete(dconn, medmask, axis=0), medmask, axis=1)
    return dconn


def _test_diffemb(aff_fpath, grad_fpath, options):
    emb,res = cgrad.comp_diffusion_embedding(aff_fpath, 
            alpha=options.alpha, 
            n_components=options.n_comp)
    cgrad.export_gradients(grad_fpath, emb, res)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute diffusion embedding of data for every subject in list')
    parser.add_argument('path_in',
            help='input filepath for group dense connectome')
    parser.add_argument('path_out', 
            help='destination filepath (contains %s marker for _emb_ and _res_ outputs)')
    parser.add_argument('-a', '--alpha', type=float, default=0.5,
            help='Diffusion embedding control parameter', dest='alpha')
    parser.add_argument('-n', '--ncomponents', type=int, default=100,
            help='Number of gradient components to return', dest='n_comp')
    args = parser.parse_args()

    gdconn_fpath = args.path_in
    grad_fpath = args.path_out

    group_diffusion_maps(gdconn_fpath, grad_fpath, args)
