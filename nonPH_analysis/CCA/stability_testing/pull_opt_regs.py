import os
import sys
import csv
import dill
import numpy as np
import scipy.stats as scst
from pathlib import Path

sys.path.append("/scratch/tyoeasley/brain_representations/src_py")

working_dir = sys.argv[1]
if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = os.path.dirname(os.path.abspath(working_dir))

date_label = working_dir.split("_paragg_")[1]

dirlist = os.listdir(working_dir)

file_list = [i for i in dirlist if ".CCA_stbl" in i]
floc_list = [os.path.join(working_dir,i) for i in file_list]

pairname_list = [i.split(".")[0] for i in file_list]
reg_vals = [None]*len(pairname_list)

for i in range(len(floc_list)):
    with open(floc_list[i],'rb') as fin:
        Var_in = dill.load(fin)

    reg_dist = [Var_in[j].lambda_opt for j in range(len(Var_in))]
    print(Path(floc_list[i]).stem)
    print(str(len(reg_dist)) + " optimal regularization hyperparameters found.")
    print(np.histogram(np.log10(reg_dist)))
    print("mean of dist. = " + str(np.mean(np.log10(reg_dist))))
    print("median of dist. = " + str(np.median(reg_dist)))
    print("mode of dist. = " + str(scst.mode(reg_dist).mode[0]))
    reg_vals[i] = scst.mode(reg_dist).mode[0]
    # reg_vals[i] = np.median(reg_dist) 

reg_list = [pairname_list,reg_vals]

out_loc = os.path.join(out_dir,"reglist_" + date_label + ".csv")

with open(out_loc,'w') as fout:
    write = csv.writer(fout)
    write.writerows(reg_list)
