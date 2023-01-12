import os
import sys
import numpy as np
from string import Template

def make_dummy_data(
        data_gendir = Template('/scratch/tyoeasley/brain_representations/ICA_reps/3T_HCP1200_MSMAll_d${dim}_ts2_Z/node_timeseries'),
        dummy_gendir= Template('/scratch/tyoeasley/brain_representations/dummy_data/ICA/d${dim}/raw_data'),
        dimlist=[15, 25, 50, 100, 200, 300],
        dummy_distr = "uniform"      # is one of "uniform" or "normal", or is otherwise a function handle that accepts a shape tuple "dims" and returns random data of shape "dims"
        ):

    distr = _switch_distr(dummy_distr=dummy_distr)

    for dim in dimlist:
        data_dir = data_gendir.substitute(dim=dim)
        dummy_dir = dummy_gendir.substitute(dim=dim)

        print('Real data read from:')
        print(data_dir)
        print('Dummy data written to:')
        print(dummy_dir)

        fpathin_list = [os.path.join(data_dir,fname) for fname in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir,fname))]

        print('')
        print('Last data path:')
        print(fpathin_list[-1])
        print('')

        for fpathin in fpathin_list:
            data = np.loadtxt(fpathin)

            # manufactures dummy data of the same shape as real data from specified distrribution with null mean and unit variance
            dummy_data = distr(data.shape)

            fname_new = 'dummy_' + os.path.basename(fpathin)
            fpath_new = os.path.join(dummy_dir, fname_new)
            np.savetxt(fpath_new, dummy_data)


def _switch_distr(dummy_distr='uniform'):
    _switcher = {
            "uniform": _unidistr,
            "normal": _normdistr
            }
    _distr = _switcher.get(dummy_distr, lambda argument: argument)
    return _distr

# returns random entries with shape=dims sampled from the uniform distrribution with null mean and unit variance
def _unidistr(dims):
    sample = 12*(np.random.random(dims) - 0.5)
    return sample

# returns random entries with shape=dims sampled from the standard normal distrribution with null mean and unit variance
def _normdistr(dims):
    sample = np.random.randn(*dims)
    return sample


if __name__=="__main__":
    make_dummy_data()
