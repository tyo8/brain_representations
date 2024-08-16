import os
import pickle
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def _load(output_fpath):
    with open(output_fpath, 'rb') as fin:
        Wphat_XY, Wp_XY, Wphat_XY_i, components, comparisons = pickel.load(fin)
        dIM_XXhat_i, Wp_XhatYhat_i, dIM_YYhat_i = components
        PDX_diag, PDY_diag, Wp_XXhat_i, Wp_YYhat_i, PDXhat_diag_i, PDYhat_diag_i = comparisons

    return 0 # SOME STUFF (but not all of it)

def _parse_output_fpath(output_fpath):
    fpath = output_fpath.replace('_L2dists','')
    Yname_all = os.path.basename(fpath).replace('Wp_hat_X_vs_Y_','').split('.')[0]
    Xname_all = os.path.basename(os.path.dirname(fpath)).replace('X_','')
    
    Yname, Yemb_dim, Ynoiselvl = _parse_name(Yname_all)
    Xname, Yemb_dim, Ynoiselvl = _parse_name(Xname_all)
