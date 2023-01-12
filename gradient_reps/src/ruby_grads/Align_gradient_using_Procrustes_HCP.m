%{ 
To run:
subject_list = [102816 103111 103212 103818 104012 104416 105014 105115 105216 105620 106016];
size_subj = size(subject_list,2)
for i = 1: size_subj
    subj = subject_list(1,i);
	Align_gradient_using_Procrustes (subj)
end
%}
	


function Align_gradient_using_Procrustes_HCP (subj)

file_path = ['/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/746sub/' num2str(subj) '/emb_' num2str(subj) '_300_grad_fast.mat'];
load(file_path) %variable name is emb
subj_emb = emb;

file_path_target = ['/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/746sub/Group/emb_Group_300_grad_fast.mat'];
load(file_path_target) %variable name is emb
target_emb = emb;

[p, Z, tr] = procrustes(target_emb, subj_emb);

realigned_mat = Z;

path_save = ['/mnt/isilon/CSC1/Yeolab/Data/HCP/HCP_derivatives/PrincipalGradients/Realign_Gradients_fast'];

full_path_save = [path_save '/' num2str(subj) '_realigned.mat'];
save(full_path_save, 'realigned_mat')


end
