import os
import sys
import csv
import numpy as np

def iter_extract_ICA(genpath = '/ceph/chpc/shared/janine_bijsterbosch/tyoeasley/brain_representations/ICA_reps/3T_HCP1200_MSMAll_d%d_ts2_Z/%s',
        dimlist = [15, 25, 50, 100, 200, 300],
        subjID_path = '/ceph/chpc/shared/janine_bijsterbosch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv', do_partial=True):

    with open(subjID_path, newline='') as fin:
        subjID_list = list(map(''.join, list(csv.reader(fin))))

    for dim in dimlist:
        data_dir = genpath % (dim, 'node_timeseries/')
        genpath_out = genpath % (dim, '%s/sub-')

        for sID in subjID_list:
            fpath = data_dir + sID + '.txt'
            ts_data = _load_data(fpath, dimval=dim)
            outpath_type = genpath_out + sID + '.csv'
            feats_from_dtseries(ts_data, outpath_type, do_partial=True)


def feats_from_dtseries(ts_data, outpath_type, do_partial=True):

    amps = np.std(ts_data, axis=1)
    netmats = np.corrcoef(ts_data)
    np.fill_diagonal(netmats,1)

    if do_partial:
        # compute partial correlation matrix of ts_data
        invcorr = np.linalg.pinv(netmats, hermitian=True)
        norms = np.diag(np.power(np.diag(invcorr), -1/2))
        partial_netmats = norms @ invcorr @ norms

    write_out(outpath_type % 'Amplitudes', amps)
    write_out(outpath_type % 'NetMats', netmats)
    if do_partial:
        partial_netmats
        write_out(outpath_type % 'partial_NMs', partial_netmats)

def _load_data(fpath, dimval=1000):
    ts_data = np.genfromtxt(fpath)
    if ts_data.shape[0] != dimval:
        ts_data = ts_data.T

    assert ts_data.shape[0]==dimval, f"Data shape {ts_data.shape} does not match expected dimension value ({dimval})"

    return ts_data


def write_out(outpath, data):
    savedir = os.path.dirname(os.path.abspath(outpath))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    np.savetxt(outpath, data)


if __name__=="__main__":
    iter_extract_ICA()
