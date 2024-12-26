import sys
import rcca
import pickle
import numpy as np

def pull_reg_val(filename):
    with open(filename,'rb') as fin:
        cca_res = pickle.load(fin)

    reg_val = cca_res.best_reg
    print("lambda_best = " + str(reg_val))

pull_reg_val(sys.argv[1])
