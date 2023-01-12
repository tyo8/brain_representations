% Set paths
addpath(genpath('/home/e.ty/Matlab_pkgs/fieldtrip'))

% Load MH score (column 18 is the first PC)
NIHaffect = load('NIHaffect.csv');
ID = load('HCP_IDs.csv');
n = length(ID);

% calculate glasser netmats
nets = zeros(360,360,n);
for s = 1:n
    fprintf('nets subject %d\n',s);
    D = ft_read_cifti(sprintf('MMP_netmats_raw/%d_REST1_LR.ptseries.nii',ID(s)));
    N1 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats_raw/%d_REST1_RL.ptseries.nii',ID(s)));
    N2 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats_raw/%d_REST2_LR.ptseries.nii',ID(s)));
    N3 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats_raw/%d_REST2_RL.ptseries.nii',ID(s)));
    N4 = corr(D.ptseries'); clear D
    netmat = mean(cat(3,N1,N2,N3,N4),3);
    nets(:,:,s) = netmat;

    netmat_name = sprintf('/scratch/tyoeasley/brain_representations/glasser/NetMats/subj-%1d.csv',ID(s));
    writematrix(netmat,netmat_name)
end
save('/scratch/tyoeasley/brain_representations/glasser/2021_R01_population/nets.mat','nets');
