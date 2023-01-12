import sys
import dill
import numpy as np
sys.path.append("/scratch/tyoeasley/brain_representations/src_py")
import lindecomp_phom as ldp

def main(argvals):
    lindecomps_path = argvals[1]
    output_basedir = argvals[2]
    
    with open(lindecomps_path,'rb') as fin:
        lindecomps = dill.load(fin)

    comp_rank = [1000, 500, 100, 50, 10, 5]

    ldp.make_bcurves_over_rank(lindecomps, comp_rank, output_basedir, hom_dims=(0,1,2))



if __name__=="__main__":
    main(sys.argv)
