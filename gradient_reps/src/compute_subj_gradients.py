import os
import sys
import csv
import h5py
import logging
import datetime
import numpy as np
import nibabel as nib
from sklearn.metrics import pairwise_distances

def seq_HCP_diffusion_maps(subjID_list_fpath, options):
    with open(subjID_list_fpath, newline='') as fin:
        tmp_list = list(csv.reader(fin))
        subjID_list = list(map(''.join, tmp_list))

    for subjID in subjID_list:
        par_HCP_diffusion_maps(subjID,options)


def par_HCP_diffusion_maps(subjID, options):
    data_list, outpath, cifti_headers = pull_subj_data(subjID,
            outpath_type=options.outpath_type)
    dconn = comp_dconn(data_list, 
            use_tanproj=options.tanproj_flag)
    aff = dconn_to_affinity(dconn, 
            prctile_thresh=options.pthresh)
    emb,res = comp_diffusion_embedding(aff, 
            alpha=options.alpha, 
            n_components=options.n_components)
    if options.tanproj_flag:
        outpath = outpath.replace('subj', 'tanproj_subj')
    
    outpath_emb = export_gradients(outpath, emb, res)
    if options.cifti_flag:
        export_cifti(outpath_emb, emb.T, headers_list=cifti_headers)
    print('DONE: ' + str(datetime.datetime.now()))



def pull_subj_data(subject_id, 
        sub_cort_start_idx = 59412,
        find_subj_path = '/scratch/tyoeasley/brain_representations/check_subj.sh',
        outpath_type = '/scratch/tyoeasley/brain_representations/gradient_reps/grad_maps/%s_subj-%s.npy'):

    print('\nStep 1: Load data and remove subcortical/cerebellar structrues')
    print(str(datetime.datetime.now()))
    outpath = outpath_type % ('%s', subject_id)
        
    #call bash script to find subject's filepath from subjID, read in filepath, and remove trailing newline '\n' from string
    fmri_fpath = os.popen(find_subj_path + ' ' + subject_id).read()[:-1]

    data_list = [None]*4
    for session in range(4):
        if session == 0:
            session_id = 'REST1_LR'
        if session == 1:
            session_id = 'REST1_RL'
        if session == 2:
            session_id = 'REST2_LR'
        if session == 3:
            session_id = 'REST2_RL'

        #read in data
        try:
            cifti_data = nib.load(fmri_fpath.replace('REST1_LR', session_id))
            cifti_headers = [cifti_data.header, cifti_data.nifti_header]
            data = cifti_data.get_fdata()
            
            # remove sub-cortical data
            data = np.delete(data, range(sub_cort_start_idx, data.shape[1]), axis=1)
        except FileNotFoundError:
            data = None

        #### debug code ####
        # print('list length: ' + str(len(data_list)))
        # print('index: ' + str(session))
        #### debug code ####
        
        data_list[session] = data
        
    data_list = [data for data in data_list if np.any(data)]    # removes None-type data entries (i.e., missing subject runs)

    return data_list, outpath, cifti_headers


#Step 2: compute correlation matrix
def comp_dconn(data_list, use_tanproj=True, subsample_flag=True, subsample_factor=0.05):
    print('\nStep 2: compute correlation matrix')
    startime = datetime.datetime.now()
    
    if use_tanproj:
        corr_data_list = [_comp_corr_mtx(data,
            subsample=False) for data in data_list]

        lapsetime = datetime.datetime.now() - startime
        print('Compute time for correlation matrix: ' + str(lapsetime))

        ## debug code ##
        # print([i.shape for i in corr_data_list])
        # exit()
        ## debug code ##

        geodist_startime = datetime.datetime.now()
        print("Finding approximate geodesic average of dense connectomes (regularizing if necessary)...")
        sys.path.append('/scratch/tyoeasley/brain_representations/src_py')
        import tanproj_avg
        dconn = tanproj_avg.reg_mean_ale(corr_data_list)

        lapsetime = datetime.datetime.now() - geodist_startime
        print('Done.')
        print('Compute time for approximate geodesic average: ' + str(lapsetime))

    else:
        print("Computing subsampled dense connectomes...")
        corr_data_list = [_comp_corr_mtx(data, 
            subsample=subsample_flag,
            subsample_factor=subsample_factor) for data in data_list]
        nan_num = np.sum(np.array(~np.isnan(corr_data_list)) + 0, axis=0)
        dconn = np.nansum(corr_data_list, axis=0)/nan_num
        
        lapsetime = datetime.datetime.now() - startime
        print('Done.')
        print('Compute time for correlation matrix: ' + str(lapsetime))
        
    print('Dconn shape: ' + str(dconn.shape))
    print('subsampling factor = ' + str(float(dconn.shape[1])/float(dconn.shape[0])))
    return dconn



def _comp_corr_mtx(data, subsample=True, subsample_factor=0.05):
    if subsample:
        N = data.shape[1]
        #set random number generator
        np.random.seed(1)
        #find subsample indices choose 5%
        subsample_idx = np.random.randint(0, N, int(N*subsample_factor))
        #compute subsampled correlation matrix
        data = data - np.mean(data,axis=0)
        varnorm_data = data/np.linalg.norm(data, axis=0)    
        corr_data = np.matmul(np.transpose(varnorm_data),varnorm_data[:,subsample_idx])
    else:
        #compute correlation matrix
        corr_data = np.corrcoef(data.T)
    
    return np.single(corr_data)


def dconn_to_affinity(dconn, prctile_thresh=90):

    startime = datetime.datetime.now()     
    perc = np.array([np.percentile(x, prctile_thresh) for x in dconn])


    for i in range(dconn.shape[0]):
        dconn[i, dconn[i,:] < perc[i]] = 0  

    lapsetime = datetime.datetime.now() - startime
    print('Thresholding (elapsed time): ' + str(lapsetime))
    print ("Minimum value is %f" % dconn.min())
    
    a = dconn<0
    b = np.sum(a,1)
    c = b!=0
    d = np.sum(c)
    print('\nStep 3: remove negative')

    print ("Negative values occur in %d rows" % d)

    dconn[dconn < 0] = 0

    print('\nStep 4: generate affinity matrix')

    startime = datetime.datetime.now()     
    aff = 1 - pairwise_distances(dconn, metric = 'cosine')

    lapsetime = datetime.datetime.now() - startime
    print('Compute time of affinity matrix: ' + str(lapsetime))
    return aff
    

def comp_diffusion_embedding(aff, alpha=0.5, n_components=10):
    from mapalign import embed
    #Generate embeddings
    print('\nStep 5: perform diffusion embedding')
    print('Shape of affinity matrix: ' + str(aff.shape))
    print('n_components = ' + str(n_components))

    startime = datetime.datetime.now()     
    embedding, results = embed.compute_diffusion_map(aff, alpha=alpha, n_components=n_components, return_result=True)

    lapsetime = datetime.datetime.now() - startime
    print('embedding done in: ' + str(lapsetime))
    return embedding, results



# NOTE: input variable 'outpath_type' should contain a '%s' substring to receive 'emb' & 'res' (i.e., results type) designations
def export_gradients(outpath_type, embedding, results=None):
    print('\nSaving results...')


    fpath_emb = outpath_type % 'emb'
    if results:
        fpath_res = outpath_type % 'res'

    savedir = os.path.dirname(os.path.abspath(fpath_emb))

    # check existence of output directory; make if nonexistent
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
        print('WARNING: created directory ' + savedir)

    np.save(fpath_emb, embedding)
    if results:
        np.save(fpath_res, results)
    

    return fpath_emb


# saves CIFTI2 version of diffusion embedding representation
# NOTE: data is expected to be in (time, brainmap) order, approximtely (1200, 9e5)
def export_cifti(outpath, data, headers_list=None, verbose=True):
    if verbose:
        print("exporting to CIFTI2 format...")
    # change file extension to .dtseries.nii
    ext = os.path.splitext(os.path.basename(outpath))[1]
    outpath = outpath.replace(ext, '.dtseries.nii')

    if not headers_list:
        headers_list = _gen_dummy_headers()

    header = headers_list[0]
    nifti_header = headers_list[1]

    # unpack CIFTI2 axes, assuming len(data.shape)==2
    ts_axis, brain_axis = [header.get_axis(i) for i in range(len(data.shape))]
    data = _zero_pad(data, n_rows=len(brain_axis))

    # change series axes of template header to match data
    new_ts_axis = nib.cifti2.cifti2_axes.SeriesAxis(0, 1, data.shape[0])
    # update cifti header
    new_header = (new_ts_axis, brain_axis)

    # create cifti template data
    new_cdata = nib.Cifti2Image(data, header=new_header, nifti_header=nifti_header)
    print("cifti data shape: " + str(new_cdata.shape))
    # export data to cifti format
    new_cdata.to_filename(outpath)
    if verbose:
        print("CIFTI2 data saved to " + outpath)


def _gen_dummy_headers(find_subj_path = '/scratch/tyoeasley/brain_representations/check_subj.sh'):
    fname = os.popen(find_subj_path + ' 100206').read()[:-1]
    cdata = nib.load(fname)
    headers_list = [cdata.header, cdata.nifti_header]
    return headers_list


def _zero_pad(data, n_rows=91282):
    zero_stack = np.zeros(data.shape[:-1] + (n_rows-data.shape[-1],))
    padded_data = np.append(data, zero_stack, axis=-1)
    return padded_data



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute diffusion embedding of data for every subject in list')
    parser.add_argument('subjects',
            help='Path to list of subject IDs expected if sequential OR subject ID number expected if parallel')
    parser.add_argument('--par', 
            dest='par_flag', default=False, action='store_const', const=True,
            help='Logical flag: to distribute or not to distribute?')
    parser.add_argument('-o','--out',
            dest='outpath_type', type=str,
            default = '/scratch/tyoeasley/brain_representations/gradient_reps/gradmaps_d100/%s_subj-%s.npy',
            help='string containing generic output destination')
    parser.add_argument('-c', '--cifti_out', 
            dest='cifti_flag', default=False, action='store_const', const=True,
            help='Logical flag: save output to cifti file?') 
    parser.add_argument('-t', '--tanproj',    
            dest='tanproj_flag', default=False, action='store_const', const=True,
            help='Logical flag: use of tangent projection to compute averages of connectivity/covariance matrices')
    parser.add_argument('--pthresh',    
            dest='pthresh', default=90, type=int,
            help='Percentile threshold for connectivity matrix thresholding (used only if tanproj_flag=False)')
    parser.add_argument('-n', '--n_components',    
            dest='n_components', default=100, type=int, 
            help='Number of components (i.e., coordinates) to keep from diffusion embedding')
    parser.add_argument('-a', '--alpha',    
            dest='alpha', default=0.5, type=float,
            help='Diffusion embedding parameter')
    parser.add_argument('-g', '--path_group_emb',
            dest='group_embedding_fpath', type=str,
            default='/scratch/tyoeasley/brain_representations/gradient_reps/emb_HCPavgS1200_gradmap.npy',
            help='Path to group embedding (for alignment)')
    parser.add_argument('-m', '--alignment_method',    
            dest='align_method', default='procrustes', type=str,
            help='Method to align subject-level diffusion embedding with group-level diffusion embedding')
    args = parser.parse_args()

    args.outpath_type = args.outpath_type.replace('gradmaps_d100', 'gradmaps_d'+str(args.n_components))

    if args.par_flag:
        subjID = args.subjects
        par_HCP_diffusion_maps(subjID, args)
    else:
        subjID_fpath = args.subjects
        seq_HCP_diffusion_maps(subjID_fpath, args)
