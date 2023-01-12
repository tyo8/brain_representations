#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:48:51 2018

@author: tnyr
"""
import sys

def Generate_10_grad_fast_HCP (subject_id):
#def Generate_10_grad_fast ():   
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
    #subject_id = '1'
    
    a_logger = logging.getLogger()
    a_logger.setLevel(logging.DEBUG)
    output_file_handler = logging.FileHandler('/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/logs/' + subject_id)
    stdout_handler = logging.StreamHandler(sys.stdout)
    a_logger.addHandler(output_file_handler)
    a_logger.addHandler(stdout_handler)
    #log_out = open('/mnt/isilon/CSC2/Yeolab/Data/ABCD/process/y0/individual_parcellation/Diffusion_Mats/logs/' + subject_id , "a")
    
    #set random number generator
    np.random.seed(1)
    #find subsample indices choose 5%
  
    subsample_idx = np.random.randint(0,59412,59412*0.05)

    sublist = open('/home/rkong/storage/ruby/data/FuncTerr_prediction/data/HCP_746sub.txt',"r")
    subname = sublist.read().splitlines()

    print(subject_id)
    for session in range(1,5):
        if session == 1:
            session_id = 'REST1_LR'
        if session == 2:
            session_id = 'REST1_RL'
        if session == 3:
            session_id = 'REST2_LR'
        if session == 4:
            session_id = 'REST2_RL'
        
        a_logger.debug(str(datetime.datetime.now()) + 'Step1: reading fmri data')
        print('Read fMRI data ... session ' + session_id)
        fmri_file = '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/individuals/' + subname[int(subject_id)-1] + '/MNINonLinear/Results/rfMRI_' + session_id + '/postprocessing/MSM_reg_wbsgrayordinatecortex/rfMRI_' + session_id + '_Atlas_MSMAll_hp2000_clean_regress.dtseries.nii'
        censor_file = '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/individuals/' + subname[int(subject_id)-1] + '/MNINonLinear/Results/rfMRI_' + session_id + '/postprocessing/MSM_reg_wbsgrayordinatecortex/scripts/rfMRI_' + session_id + '_FD0.2_DV75_censoring.txt'

        #obtain censor mask
        censor_data = open(censor_file,'r')
        censor_data = censor_data.read().splitlines()
        censor_mask = [i for i, val in enumerate(censor_data) if not val]

        #read in data
        data = nib.load(fmri_file).get_data() 
        data = np.transpose(np.reshape(data,(1200,96854)))
        data = np.delete(data,range(64984,96854),axis=0)

        a_logger.debug(str(datetime.datetime.now()) + 'Step1: excluding censored frames')

        #remove censored frames
        data = np.delete(data, censor_mask, axis=1)

        a_logger.debug(str(datetime.datetime.now()) + 'Step1: excluding medial wall')
        
        #remove medial wall
        medial_mask_file = open('/mnt/isilon/CSC1/Yeolab/Users/rkong/ruby/data/FuncTerr_prediction/code/PrincipalGradient/maskfslr32.txt',"r")
        medial_mask_data = medial_mask_file.read().splitlines()
        medial_mask_data = np.array(medial_mask_data, dtype=float)
        medial_mask = [i for i, val in enumerate(medial_mask_data) if val == 1]
        data = np.transpose(np.delete(data, medial_mask, axis=0))


        #compute correlation matrix
        a_logger.debug(str(datetime.datetime.now()) + 'Step2: compute correlation matrix')
        startime = datetime.datetime.now()
        data = data - np.mean(data,axis=0)
        data = data/np.linalg.norm(data, axis=0)    
        data = np.matmul(np.transpose(data),data[:,subsample_idx])
        endtime = datetime.datetime.now() 
        lapsetime = endtime - startime
        temp_str = 'Compute correlation matrix:' + str(lapsetime)
        print(temp_str)
        
        if session == 1:
            dcon = data
            nan_num = np.array(~np.isnan(data)) + 0
        else:
            #data = np.where(np.isnan(data), dcon, np.nan_to_num(data))
            #dcon = dcon + data
            dcon = np.nansum(np.array([data,dcon]),0)
            nan_num = np.array(~np.isnan(data)) + 0 + nan_num

    #average correlation matrix
    a_logger.debug(str(datetime.datetime.now()) + 'Step2: average correlation matrix')
    dcon = dcon/nan_num 
    #dcon = dcon/4
    #dcon[np.isnan(dcon)] = 0
    del data
    
    N = dcon.shape[0]
    a_logger.debug(str(datetime.datetime.now()) + 'Step3: choose top 90%')
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
    aff = 1 - pairwise_distances(dcon, metric = 'cosine')

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
    emb, res = embed.compute_diffusion_map(aff, alpha = 0.5, n_components=10, return_result=True)

    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    temp_str = 'embedding done:' + str(lapsetime)
    print(temp_str)   

    a_logger.debug(str(datetime.datetime.now()) + 'Step6: save out results')
    path_name_results = '/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/746sub/'+ subject_id + '/'

    file_emb = 'emb_' + subject_id + '_10_grad_fast.npy'
    file_res = 'res_' + subject_id + '_10_grad_fast.npy'
    full_path_emb = path_name_results + file_emb
    full_path_res = path_name_results + file_res

    startime = datetime.datetime.now() 
    np.save(full_path_emb, emb)
    np.save(full_path_res, res)
    
    sio.savemat(path_name_results + 'emb_' + subject_id + '_10_grad_fast.mat', dict(emb = emb))
    sio.savemat(path_name_results + 'res_' + subject_id + '_10_grad_fast.mat', dict(res = res))
    
    endtime = datetime.datetime.now() 
    lapsetime = endtime - startime
    print(lapsetime)

    a_logger.debug(str(datetime.datetime.now()) + 'DONE')
    #sys.stdout.close()
Generate_10_grad_fast_HCP (sys.argv[1])
#Generate_10_grad_fast ()




