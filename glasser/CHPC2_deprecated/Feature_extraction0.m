% Set paths
addpath(genpath('/home/janine/Matlab/fieldtrip-20190819/'))

% Load MH score (column 18 is the first PC)
NIHaffect = load('NIHaffect.csv');
ID = load('HCP_IDs.csv');

% calculate glasser netmats
nets = zeros(360,360,1000);
for s = 1:1000
    fprintf('nets subject %d\n',s);
    D = ft_read_cifti(sprintf('MMP_netmats/%d_REST1_LR.ptseries.nii',ID(s)));
    N1 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/%d_REST1_RL.ptseries.nii',ID(s)));
    N2 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/%d_REST2_LR.ptseries.nii',ID(s)));
    N3 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/%d_REST2_RL.ptseries.nii',ID(s)));
    N4 = corr(D.ptseries'); clear D
    nets(:,:,s) = mean(cat(3,N1,N2,N3,N4),3); clear N1 N2 N3 N4 D
end
save('nets.mat','nets');

% calculate ICA amplitudes
amps = zeros(300,1000);
for s = 1:1000
    fprintf('amps subject %d\n',s);
    D = load(sprintf('ICA_amplitudes/%d.txt',ID(s)));
    amps(:,s) = std(D); clear D
end
save('amps.mat','amps');
