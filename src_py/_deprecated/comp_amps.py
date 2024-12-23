import os
import re
import sys
import csv
import numpy as np

sIDs_fname="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/HCP_IDs_all.csv"

def main(parent_dir, out_dir, dim_subdirs=True, subj_flist=sIDs_fname):
    with open(subj_flist,newline='') as fin:
        subj_List = list(csv.reader(fin))

    subj_list = list(map(''.join, subj_List))

    if dim_subdirs:
        dim_nums = [subdir for subdir in os.listdir(parent_dir)
                if subdir.startswith('d')]
        for i in dim_nums:
            parent_dir_i = os.path.join(parent_dir,i)
            out_dir_i = os.path.join(out_dir,i)

            calc_and_save_amps(parent_dir_i, out_dir_i, subj_list)
    else:
        calc_and_save_amps(parent_dir, out_dir, subj_list)



def calc_and_save_amps(parent_dir, out_dir, subj_list):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print("Warning: created Amplitudes directory " + out_dir)

    for s in subj_list:
        regex_s = "*" + s + "*.csv"
        filelist_s = [f for f in os.listdir(parent_dir) if re.match(regex_s, f)]
        fileloc_s = [os.path.join(parent_dir, f) for f in filelist_s]
        amps_f_s = [np.std(np.loadtxt(f)) for f in fileloc_s]
        amps_s = np.concatenate(amps_f_s)

        floc_out = os.path.join(out_dir, 'subj-' + s + '.csv')
        np.savetxt(fname_out, amps_s)



if __name__=="__main__":
    parent_dir = sys.argv[1]
    out_dir = sys.argv[2]
    if len(sys.argv > 3):
        dim_subdirs = sys.argv[3]
        if len(sys.argv > 4):
            subj_flist = sys.argv[4]
    main(parent_dir, out_dir, dim_subdirs=dim_subdirs, subj_flist=subj_flist):
