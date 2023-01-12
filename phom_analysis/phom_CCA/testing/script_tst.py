import sys
sys.path.append("/scratch/tyoeasley/brain_representations/src_py")
import numpy as np
import SubjSubjHomologies as SSH

def main(argvals):
    path_to_tst = argvals[1]
    dist_mtx = np.load(path_to_tst, allow_pickle=True)

    dist_mtx = np.stack([dist_mtx]*9, axis=0)

    path_out = '/scratch/tyoeasley/brain_representations/phom_analysis/phom_CCA/testing/phoms_tst.npy'
    hom_dims = (0,1,2)

    if len(argvals) > 2:
        path_out= argvals[2]
        if len(argvals) > 3:
            hom_dims = argvals[3]

    phom = SSH.compute_phom(dist_mtx, hom_dims=hom_dims)
    np.save(path_out, phom)

if __name__=="__main__":
    main(sys.argv)
