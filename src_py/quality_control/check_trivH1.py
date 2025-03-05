import os
import glob
import argparse
import numpy as np

triv_fpath = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/trivial_H1/trivial_first_homology.txt"
search_pattern = "/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/full-scale-expmt/within_*/*_*/phom_data_*_dists/phom_X.txt"
H1line = 'persistent homology intervals in dim 1:'

base_dir="/ceph/chpc/shared/janine_bijsterbosch_group/tyoeasley/brain_representations/phom_analysis/full-scale-expmt"

def check_discrete_metric(triv_path, output = "out.txt", thresh=0.95):
    with open(triv_path, 'r') as fin:
        trivnames = fin.read().split('\n')

    dist_mtxs = {}
    for name in trivnames:
        var = '_'.join(name.split('_')[:-1])
        pattern = os.path.join(base_dir,f"within_*/*_*/{name}*.txt")
        try:
            fname = glob.glob(pattern)[0]
        except:
            print(f"found no matches to {pattern}")
            continue

        dist_mtxs[var] = np.loadtxt(fname)
        # print(f"Matched \'{name}\' at \'{fname}\' and recording in dictionary \'dist_mtxs\' as \'{var}\'")

    with open(output, 'w') as fout:
        for key in list(dist_mtxs.keys()):
            mtx = dist_mtxs[key]
            vals = _triu_vals(mtx)
            fout.write("**********\n")
            fout.write(f"var \'{key}\' has shape {mtx.shape}\n")
            fout.write(f"and values: \n{np.histogram(vals)}\n")
            fout.write(f"with upper-triangular mean of {np.mean(vals)}\n\n")

    rerun_path = os.path.join(os.path.dirname(triv_path), os.path.basename(triv_path).replace("trivial","rerun"))
    rerun_count=0
    with open(rerun_path, 'w') as fout:
        for key in list(dist_mtxs.keys()):
            mtx = dist_mtxs[key]
            vals = _triu_vals(mtx)
            if np.mean(vals) <= thresh:
                fout.write(f"{key}\n")
                rerun_count += 1

    print(f"Recommending {rerun_count} phom_X re-computations after thresholding the discrete metric at 1>r>{thresh} (see \'{rerun_path}\')")

    print(f"distance matrices summaries for H1-trivial barcodes written to file (check for approximate discrete metric): \n\'{output}\'")
    return None


def check_trivH1(pattern, out=triv_fpath, line_val=H1line):
    fpath_list = glob.glob(pattern)
    fpath_list.sort()

    triv_h1_list = []
    triv_BR_list = []
    for fpath in fpath_list:
        with open(fpath, 'r') as fin:
            try:
                last_line = fin.readlines()[-1]
            except:
                # print(f"read failed for: \n{fpath}")
                triv_BR_list.append(_get_name(fpath))
        if line_val in last_line:
            triv_h1_list.append(_get_name(fpath))

    print(f"Found {len(triv_BR_list)} empty, failed, or nonexistent persistence outputs.")
    print(f"Found {len(triv_h1_list)} persistence barcodes with trivial H1.")

    with open(out, 'w') as fout:
        for name in triv_h1_list:
            fout.write(f"{name}\n")
    with open(out.replace("first_homology","brainreps"), 'w') as fout:
        for name in triv_BR_list:
            fout.write(f"{name}\n")
    return None


def _get_name(fullpath):
    dirname = os.path.basename(os.path.dirname(fullpath))
    name = dirname.replace("phom_data_","").replace("_dists",'')
    return name

def _triu_vals(A):
    n = min(A.shape)
    vals = A[np.triu_indices(n,1)]
    return vals


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "-i",
        "--search_pattern",
        type=str,
        default=search_pattern,
        help=""
    )
    parser.add_argument(
        "-o",
        "--results_fpath",
        type=str,
        default=os.path.join(os.getcwd(),"BR_metric_distribution_summaries.txt"),
        help=""
    )
    parser.add_argument(
        "-l",
        "--trivial_list_fpath",
        type=str,
        default=triv_fpath,
        help=""
    )
    args = parser.parse_args()
    check_trivH1(args.search_pattern, out=args.trivial_list_fpath, line_val=H1line)
    check_discrete_metric(args.trivial_list_fpath, output = args.results_fpath)
