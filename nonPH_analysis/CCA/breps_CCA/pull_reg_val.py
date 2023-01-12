import sys
import rcca
import pickle
import numpy as np

def pull_reg_val(filename,fullstring = True):
    with open(filename,'rb') as fin:
        cca_res = pickle.load(fin)

    reg_val = cca_res.best_reg
    min_cancorr = np.amin(cca_res.cancorrs)
    max_cancorr = np.amax(cca_res.cancorrs)
    gmean_cancorr = np.exp(np.mean(np.log(cca_res.cancorrs)))
    if fullstring:
        print("lambda_best = " + str(reg_val))
        print("min can. corr = " + str(min_cancorr))
        print("max can. corr = " + str(max_cancorr))
        print("gmean can. corr = " + str(gmean_cancorr))
    else:
        print(str(reg_val))
        print(str(min_cancorr))

pull_reg_val(sys.argv[1])
