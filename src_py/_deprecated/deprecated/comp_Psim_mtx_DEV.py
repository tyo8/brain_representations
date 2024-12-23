# receives path to .csv file containing list of subject data file names
# outputs a subject-by-subject matrix of Pearson correlation coefficients

import numpy as np
import csv
import parse_neurodata as parse
import os

def comp_dist_mtx(fname_in,fname_out):
    with open(fname_in,newline='') as fin:
        subj_lists = list(csv.reader(fin))      # list of lists of subject filenames

    subj_list = list(map(''.join,subj_lists))   # list of strings of subject filenames
    Psim_mtx = comp_Psim_vals(subj_list)
    np.savetxt(fname_out,Psim_mtx,delimiter=",")
    return Psim_mtx


def comp_Psim(data1,data2):
    sim_mtx = np.corrcoef(data1,data2)
    simval = sim_mtx[0,1]
    return simval


# the function comp_Psim_vals could be rewritten to recursively check and split lists that try to load more than 1TB of data

def comp_Psim_vals_old(subj_list):
    n_subj = len(subj_list)
    Psim_mtx = np.zeros((n_subj,n_subj))

    if n_subj > 10**4:
        for i in range(n_subj):
            subj1_fname = subj_list[i]
            data1 = parse.parse_fname(subj1_fname)
            for j in range(i+1,n_subj):
                subj2_fname = subj_list[j]
                data2 = parse.parse_fname(subj2_fname)
                simval = comp_Psim(data1,data2)
                Psim_mtx[i,j] = simval
    else:
        data_list = ['']*n_subj
        for i in range(n_subj):
            data_list[i] = parse.parse_fname(subj_list[i])

        for i in range(n_subj):
            for j in range(i+1,n_subj):
                simval = comp_Psim(data_list[i],data_list[j])
                Psim_mtx[i,j] = simval

    Psim_mtx = Psim_mtx + np.identity(len(subj_list)) + np.matrix.transpose(Psim_mtx)
    Psim_mtx = 1 - Psim_mtx
    return Psim_mtx


def check_lists(subj_list,threshold = 2**37)
    fsize = meas_fsize(subj_list)

    if fsize < threshold


def meas_fsize(subj_list):
    n_subj = len(subj_list)

    fsizes = []
    for fname in subj_list:
        fsizes.append(os.stat(fname).st_size())

    fsize = np.array(fsizes).sum()
    return fsize


def psim_from_subj_list(subj_list1,subj_list2):
    n1 = len(subj_list1)
    n2 = len(subj_list2)
    dlist1 = []
    dlist2 = []
    for i in range(n1):
        dlist1.append(parse.parse_fname(subj_list1[i]))

    for i in range(n2):
        dlist2.append(parse.parse_fname(subj_list2[i]))

    psim = psim_from_data(dlist1,dlist2)
    return psim


def psim_from_data(data1,data2):
    n1 = len(data1)
    n2 = len(data2)

    psim_mtx = np.zeros((n1,n2))

    for i in range(n1):
        for j in range(n2):
            psim_mtx[i,j] = comp_psim(data1[i],data2[j])

    return psim_mtx

def split_list(subj_list):

