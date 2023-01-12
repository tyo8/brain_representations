import os
import sys
import csv
import numpy as np

def iter_extract_yeo(genpath = '/scratch/tyoeasley/brain_representations/yeo/ext_ptseries/wb_cp/%s_%s.csv',
        subjID_path = '/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv', do_partial=True):

    with open(subjID_path, newline='') as fin:
        subjID_list = list(map(''.join, list(csv.reader(fin))))

    label_list = ['REST1_LR', 'REST1_RL', 'REST2_LR', 'REST2_RL']

    genpath_out = '/scratch/tyoeasley/brain_representations/yeo/%s/sub-%s.csv'
    
    for sID in subjID_list:
        runs_data = [None]*len(label_list)
        for i in range(len(label_list)):
            try:
                fpath = genpath % (sID, label_list[i])
                runs_data[i] = _load_data(fpath)
            except IOError:
                err_log = open('missing_subj_data.csv','a')
                err_log.write(fpath+'\n')
                err_log.close()
                runs_data.pop(-1)

        ts_data = np.concatenate(runs_data, axis=1)
        outpath_type = genpath_out % ('%s', sID) 
        feats_from_dtseries(ts_data, outpath_type, do_partial=True)


def feats_from_dtseries(ts_data, outpath_type, do_partial=True):
    print('Shape of timeseries data: ' + str(ts_data.shape))
    amps = np.std(ts_data, axis=1)
    netmats = np.corrcoef(ts_data)

    if do_partial:
        partial_netmats = _comp_partial_netmats(ts_data)
        # compute partial correlation matrix of ts_data

    write_out(outpath_type % 'Amplitudes', amps)
    write_out(outpath_type % 'NetMats', netmats)
    if do_partial:
        partial_netmats
        write_out(outpath_type % 'partial_NMs', partial_netmats)


def _comp_partial_netmats(data):
    C = np.cov(data)
    pcorr = np.linalg.pinv(C, hermitian=True)
    return pcorr


def _load_data(fpath):
    ts_data = np.genfromtxt(fpath)
    if ts_data.shape[0] > ts_data.shape[1]:
        ts_data = ts_data.T

    return ts_data


def write_out(outpath, data):
    savedir = os.path.dirname(os.path.abspath(outpath))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    np.savetxt(outpath, data)


if __name__=="__main__":
    iter_extract_yeo()
