#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:48:51 2018

@author: tnyr
"""
import sys

def Generate_group_grad():
    
    import numpy as np
    import nibabel as nib # git clone --branch enh/cifti2 https://github.com/satra/nibabel.git
    from sklearn.metrics import pairwise_distances
    import datetime
    import h5py
    import scipy.io as sio
    import nibabel as nib
    import os
    import sys
    import logging
    
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.DEBUG)
    output_file_handler = logging.FileHandler('/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/logs/group-level')
    stdout_handler = logging.StreamHandler(sys.stdout)
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    #sys.stdout = open('/mnt/isilon/CSC2/Yeolab/Data/ABCD/process/y0/individual_parcellation/Diffusion_Mats/logs/group_level', "w")
    
    a_logger.debug(str(datetime.datetime.now()) + 'Step1: read in correlation matrix')
    #set random number generator
    np.random.seed(1)
    #find subsample indices choose 5%
  
    subsample_idx = np.random.randint(0,59412,59412*0.05)

    dcon = np.load('/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/HCP_pred_gradients/FC_matrices/746_Group_FC/Group_FC.npy')
    
    dcon = dcon[:,subsample_idx]
    
    N = dcon.shape[0]
    a_logger.debug(str(datetime.datetime.now()) + 'Step2: choose top 90%')

    startime = datetime.datetime.now()     
    perc = np.array([np.percentile(x, 90) for x in dcon])

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'Matrix size:' + str(N)
    print(temp_str)
    temp_str = 'Thresholding:' + str(lapsetime)
    print(temp_str)

    startime = datetime.datetime.now()     
    for i in range(dcon.shape[0]):
        #print ("Row %d" % i)
        dcon[i, dcon[i,:] < perc[i]] = 0  

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'Set below threshold values to be 0' + str(lapsetime)
    print(temp_str)   
    
    print ("Minimum value is %f" % dcon.min())
    
    a = dcon<0
    b = np.sum(a,1)
    c = b!=0
    d = np.sum(c)
    
    a_logger.debug(str(datetime.datetime.now()) + 'Step3: remove negative')

    print ("Negative values occur in %d rows" % d)

    startime = datetime.datetime.now()     
    dcon[dcon < 0] = 0

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'Set negative values to be 0:' + str(lapsetime)
    print(temp_str)   
    a_logger.debug(str(datetime.datetime.now()) + 'Step4: generate distance matrix')

    startime = datetime.datetime.now()     
    dcon = 1 - pairwise_distances(dcon, metric = 'cosine')

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'Compute distance matrix:' + str(lapsetime)
    print(temp_str)   
    
    #file_name_aff = 'aff_' + subject_id +'.npy'
    #full_path_aff = path_name_results + file_name_aff
    #np.save(full_path_aff, aff)
    
    a_logger.debug(str(datetime.datetime.now()) + 'Step5: perform embedding')
   
    #Generate embeddings
    import sys

    sys.path.append(os.environ['CBIG_CODE_DIR'] + '/external_packages/python/mapalign-master/')

    from mapalign import embed    

    startime = datetime.datetime.now()     
    emb, res = embed.compute_diffusion_map(dcon, alpha = 0.5, n_components=300, return_result=True)

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'Part 6:' + str(lapsetime)
    print(temp_str)   

    a_logger.debug(str(datetime.datetime.now()) + 'Step6: save out results')
  
    path_name_results = '/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/746sub/Group/'

    file_emb = 'emb_Group_300_grad_fast.npy'
    file_res = 'res_Group_300_grad_fast.npy'
    full_path_emb = path_name_results + file_emb
    full_path_res = path_name_results + file_res

    startime = datetime.datetime.now() 
    np.save(full_path_emb, emb)
    np.save(full_path_res, res)

    sio.savemat(path_name_results + 'emb_Group_300_grad_fast.mat', dict(emb = emb))
    sio.savemat(path_name_results + 'res_Group_300_grad_fast.mat', dict(res = res))

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    print(lapsetime)
    a_logger.debug(str(datetime.datetime.now()) + 'DONE')

    #sys.stdout.close()
Generate_group_grad ()




