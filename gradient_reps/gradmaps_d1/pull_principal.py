import sys
import numpy as np

def pull_principal(emb_fpath, out_fpath):
    emb = np.load(emb_fpath)
    principal = emb[:,1]
    np.save(out_fpath, principal)

if __name__=="__main__":
    emb_fpath = sys.argv[1]
    out_fpath = sys.argv[2]
    pull_principal(emb_fpath, out_fpath)
