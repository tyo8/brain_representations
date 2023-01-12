import os
import sys
import numpy as np

#DEPRECATED: n_rows=96854

def make_emb_precifti(emb_fpath, outname, mask_flag=False,
        mask_fpath='/scratch/tyoeasley/brain_representations/gradient_reps/src/comp_HCP_gradients/maskfslr32.txt', 
        outdir='/scratch/tyoeasley/brain_representations/gradient_reps/viz_maps', 
        n_rows=91282):
    ext = os.path.splitext(os.path.basename(emb_fpath))[1]
    if ext == '.txt':
        emb = np.loadtxt(emb_fpath)
    elif ext == '.npy':
        emb = np.load(emb_fpath)
    else:
        raise Exception('Unrecognized file extension.')

    if mask_flag:
        emb_precifti = masked_to_precifti(mask_fpath, emb, n_rows=n_rows)
    else:
        emb_precifti = zero_pad(emb, n_rows=n_rows)
    outpath = os.path.join(outdir, outname)
    np.savetxt(outpath, emb_precifti)
    

def zero_pad(data, n_rows=91282):
    zero_stack = np.zeros(data.shape[:-1] + (n_rows-data.shape[-1],))
    padded_data = np.append(data, zero_stack, axis=-1)
    return padded_data


def masked_to_precifti(mask_fpath, data, n_rows=91282):
    with open(mask_fpath, 'r') as fin:
        mask = np.array(fin.read().splitlines(), dtype=int)
    
    cortical_mask = [bool(1-i) for i in mask]

    if len(data.shape)==1:
        cortical_data = np.zeros(mask.shape)
        cortical_data[cortical_mask] = data

    else:
        # NOTE: assumes grayordinates are stored in *LAST* dimension of data array (if multidimensional)!
        cortical_data = np.zeros(data.shape[:-1] + mask.shape)
        cortical_data[:,cortical_mask] = data

    zero_stack = np.zeros(cortical_data.shape[:-1] + (n_rows-len(mask),))
    precifti_data = np.append(cortical_data, zero_stack, axis=-1)

    return precifti_data


if __name__=='__main__':
    emb_fpath = sys.argv[1]
    outname = sys.argv[2]
    masked_to_precifti(emb_fpath, outname)
