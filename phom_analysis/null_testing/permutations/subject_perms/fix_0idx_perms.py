import os
import sys
import numpy as np

def enforce_0idx(idx_path):
    idx = np.loadtxt(idx_path).astype(int)

    counter=0
    if idx.min()==1:
        idx -= 1
        counter+=1

    np.savetxt(idx_path, idx, fmt='%i')

    return counter

if __name__=="__main__":
    idx_dir = sys.argv[1]
    pattern = sys.argv[2]

    print(f"Checking files containing \"{pattern}\" in directory \'{idx_dir}\' for compliance with 0-indexing convention...")

    idx_paths = [path for path in os.listdir(idx_dir) if pattern in path]
    counter=0
    for path in idx_paths:
        counter += enforce_0idx(path)

    print("Done. Corrected {counter} of {len(idx_paths)} pattern-matched files in \'{idx_dir}\'.")
