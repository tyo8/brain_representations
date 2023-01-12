import sys
import numpy as np


if __name__=="__main__":
    data_fname = sys.argv[1]
    if data_fname.endswith('.npy'):
        data = np.load(data_fname)
    elif data_fname.endswith('.txt'):
        data = np.loadtxt(data_fname)

    print('Data has shape ' + str(data.shape))
    numnonzero = np.count_nonzero(np.diagonal(data))
    if numnonzero:
        print(data_fname + ' has ' + str(numnonzero) + ' nonzero entries along diagonal')
