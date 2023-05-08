import os
import sys
import csv
import numpy as np
from string import Template

def iter_extract_dummy(genpath = Template('/scratch/tyoeasley/brain_representations/dummy_data/${emulator}/d${dim}/${rel_path}'),
        emulator = 'ICA',
        dimlist = [15, 25, 50, 100, 200, 300],
        subjID_path = '/scratch/tyoeasley/HCPsubj_subsets/HCP_IDs_all.csv', do_partial=True):

    print('Emulating data of type: ' + emulator)
    
    with open(subjID_path, newline='') as fin:
        subjID_list = list(map(''.join, list(csv.reader(fin))))

    for dim in dimlist:
        data_dir = genpath.substitute(emulator=emulator, dim=dim, rel_path='raw_data/dummy_')

        for sID in subjID_list:
            fpath = data_dir + sID + '.txt'
            ts_data = _load_data(fpath, dimval=dim)
            outpath_type = Template(genpath.substitute(emulator=emulator, dim=dim, 
                rel_path='${feat_type}/dummy_sub-' + sID + '.txt'))
            feats_from_dtseries(emulator, ts_data, outpath_type, do_partial=True)


def feats_from_dtseries(emulator, ts_data, outpath_type, do_partial=True):

    amps = np.std(ts_data, axis=1)
    netmats = np.corrcoef(ts_data)

    if do_partial:
        partial_netmats = _comp_partial_netmats(ts_data)
        # compute partial correlation matrix of ts_data

    write_out(outpath_type.substitute(feat_type='Amplitudes'), amps)
    write_out(outpath_type.substitute(feat_type='NetMats'), netmats)
    if do_partial:
        partial_netmats = _comp_partial_netmats(ts_data)
        write_out(outpath_type.substitute(feat_type='partial_NMs'), partial_netmats)


def _comp_partial_netmats(data):
    C = np.cov(data)
    pcorr = np.linalg.pinv(C, hermitian=True)
    return pcorr

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
    iter_extract_dummy()
