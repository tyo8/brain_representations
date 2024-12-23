import sys
import csv
import HCP_utils
import lindecomp_brainrep as ld_br


dataset_fname = sys.argv[1]
output_basedir = sys.argv[2]
namelist_path = sys.argv[3]
reglist_path = sys.argv[4]
decomp_method = sys.argv[5]

reps = HCP_utils.load_reps(dataset_fname)
namelist = HCP_utils.extract_namelist(namelist_path)
with open(reglist_path,'r') as fin:
    reglist = list(csv.reader(fin))
    ## debug code
    print(reglist)

ld_br.pairwise_lindecomp2(output_basedir, reps, namelist, reglist=reglist, decomp_method=decomp_method)
