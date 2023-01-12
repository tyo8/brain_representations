import sys
import csv
import numpy as np
from scipy.spatial import procrustes
from compute_subj_gradients import export_gradients


def align_all_subjs(subjID_fpath, cifti_flag=False,
        group_embedding_fpath = '/scratch/tyoeasley/brain_representations/gradmaps_d100/emb_HCPavgS1200.npy',
        subjpath_type = '/scratch/tyoeasley/brain_representations/gradient_reps/gradmaps_d100/%s_subj-%s.npy'):

    outpath_type = subjpath_type.replace('.npy','_aligned.npy')

    with open(subjID_fpath, newline='') as fin:
        tmp_list = list(csv.reader(fin))
        subjID_list = list(map(''.join, tmp_list))

    group_embedding = np.single(np.load(group_embedding_fpath))
    for subjID in subjID_list:
        subj_embedding = np.load(subjpath_type % ('emb',subjID))
        emb_aligned = align_to_group(group_embedding, subj_embedding)
        outpath_emb = export_gradients(outpath_type % ('%s', subjID), emb_aligned)

        if cifti_flag:
            from compute_subj_gradients import export_cifti
            export_cifti(outpath_emb, emb_aligned.T)
        print('subj embedding '+str(subjID)+' successfully aligned and written.')


def align_to_group(group_embedding, subj_emb, method='procrustes'):
    if method=='procrustes':
        std_gpemb, aligned_emb, disparity = procrustes(group_embedding, subj_emb)
        print('Disparity val. = ' + str(disparity))
    else:
        assert (1 < 0), 'ERROR: ' + method + ' method not implemented!'
        # potentially implement hungarian method?
    return aligned_emb



if __name__=="__main__":
    subjIDs_fpath = sys.argv[1]
    align_all_subjs(subjIDs_fpath, 
            group_embedding_fpath=sys.argv[2],
            subjpath_type=sys.argv[3], 
            cifti_flag=sys.argv[4])

