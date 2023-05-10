import os
import sys
import scipy
import argparse
import numpy as np

################################################################################################################
def compare_distributions(datavec1, datavec2):
    KS_res = scipy.stats.kstest(datavec1, datavec2, method='exact', alternative='two-sided')

    pdf1 = np.histogram(datavec1)[0]/len(datavec1)
    pdf2 = np.histogram(datavec2)[0]/len(datavec2)
    JS_dist = scipy.spatial.distance.jensenshannon(pdf1, pdf2, base=2)

    return KS_res.pval, JS_dist

def pad_matched_B1(datavec, n_resamps=1000):
    datavec = np.concatenate( (datavec, np.zeros(n_resamps - len(datavec))), axis=0 )
    return datavec

def parse_fpath(fpath):

    varname = os.basename(os.dirname(fpath)).replace('phom_data_','')

    is_B1dist=False
    if "n_matched" in fpath:
        is_B1dist=True

    return varname, is_B1dist
################################################################################################################


################################################################################################################
# parses input, streams output
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate subindex samplings and their unique tags"
    )
    parser.add_argument(
        "-p", "--fpath1", type=str, default=str, help="filepath to first distribution in pairwise comparison"
    )
    parser.add_argument(
        "-q", "--fpath2", type=str, default=str, help="filepath to second distribution in pairwise comparison"
    )

    args = parser.parse_args()

    varname_1, is_B1dist_1 = parse_fpath(args.fpath1)
    varname_2, is_B1dist_2 = parse_fpath(args.fpath2)

    if is_B1dist_1 and is_B1dist_2:
        datavec1 = pad_matched_B1(np.genfromtxt(args.fpath1))
        datavec2 = pad_matched_B1(np.genfromtxt(args.fpath2))
    else:
        datavec1 = np.genfromtxt(args.fpath1)
        datavec2 = np.genfromtxt(args.fpath2)

    KS_pval, JS_dist = compare_distributions(datavec1, datavec2)

