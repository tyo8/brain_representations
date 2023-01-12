import os
import sys

src_dir = sys.argv[1]
agg_dir = sys.argv[2]
out_dir = sys.argv[3]

sys.path.append(src_dir)
import lindecomp_stability as ld_stb

print(agg_dir)
decomp_vals = ld_stb.aggregate_decomps([agg_dir],del_dir=False)

## debug code:
print(len(decomp_vals))
print(len(decomp_vals[0]))

saveloc = ld_stb.summarize_iters(decomp_vals, out_dir, decomp_method = "CCA")
print("Summary of CCA stability iteration saved to " + saveloc)
