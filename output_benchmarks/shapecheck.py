import sys
import numpy as np

if __name__=="__main__":
    data_fname = sys.argv[1]
    data = np.load(data_fname)
    print('Data has shape ' + str(data.shape))
