import os
import sys

src_dir = "/scratch/tyoeasley/brain_representations/src_py"
sys.path.append(src_dir)

import lindecomp_stabiliyt as ld_stb

def main(argvals):
    agg_dir = argvals[1]
    out_dir = argvals[2]

    print(agg_dir)
    decomp_vals = ld_stb.aggregate_decomps([agg_dir],del_dir=False)

    ## debug code:
    print(len(decomp_vals))
    print(len(decomp_vals[0]))

    saveloc = ld_stb.summarize_iters(decomp_vals, out_dir, decomp_method = "CCA")
    print("Summary of CCA stability iteration saved to " + saveloc)

if __name__=="__main__":
    main(sys.argv)
