import os
import sys
import csv
import dill
import numpy as np
import brainrep as br

# add parent directory to path instead of using relative import, which fails in command line use case
sys.path.append("/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/src_py")
import HCP_utils as hutils

def pull_stbl_cancorrs(input_paragg_dir,output_cancorr_dir,namelist_path,reglist_path,
        decomp_method = 'CCA'):
    with open(reglist_path,newline='') as fin:
        reglist = list(csv.reader(fin))

    namelist = hutils.extract_namelist(namelist_path)
    n_reps = len(namelist)
    ext_in = "." + decomp_method + "_stbl"

    for i in range(n_reps):
        for j in range(i+1,n_reps):
            pairname = namelist[i] + "_and_" + namelist[j] 
            reg_val = br.find_reg_val(reglist,namelist[i],namelist[j])

            input_fname = pairname + ext_in 
            input_floc = os.path.join(input_paragg_dir,input_fname)
            with open(input_floc,'rb') as fin:
                stbl_vars = dill.load(fin)

            cancorrs = corrs_from_vars(stbl_vars,reg_val)

            output_fname = pairname + "_cancorr_data.csv"
            output_floc = os.path.join(output_cancorr_dir,output_fname)
            print(pairname)
            with open(output_floc,'w') as fout:
                write = csv.writer(fout)
                write.writerow(cancorrs)

def corrs_from_vars(stbl_vars,reg_val):
    cancorrs_all = np.asarray([i.can_corrs for i in stbl_vars if (i.lambda_opt == reg_val)])
    cancorrs = np.median(cancorrs_all,axis=0)
    print("Maximum per-component standard deviation over same-reg realizations: " + str(np.amax(np.std(cancorrs_all,axis=0))))
    # print(cancorrs_all.shape)
    # print(cancorrs.shape)
    return cancorrs


def pull_res_cancorrs(res_dir):
    varnames_all = os.listdir(res_dir)
    varin_names = [i for i in varnames_all if ".CCA_res" in i]
    varout_names = [i.split(".")[0] + "_cancorr_data.csv" for i in varin_names]
    varin_locs = [os.path.join(res_dir,i) for i in varin_names]
    varout_locs = [os.path.join(res_dir,i) for i in varout_names]
    
    for i in range(len(varin_locs)):
        with open(varin_locs[i],'rb') as fin:
            resvars = dill.load(fin)

        cancorrs = resvars.cancorrs
        with open(varout_locs[i],'w') as fout:
            write = csv.writer(fout)
            write.writerow(cancorrs)


if __name__=="__main__":
    input_paragg_dir = sys.argv[1]
    output_cancorr_dir = sys.argv[2]
    namelist_path = sys.argv[3]
    reglist_path = sys.argv[4]
    pull_stbl_cancorrs(input_paragg_dir, output_cancorr_dir, namelist_path, reglist_path)
