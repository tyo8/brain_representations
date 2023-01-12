clear all; close all; clc

% Set paths
addpath(genpath('/home/janine/Matlab/fieldtrip-20190819/'))
ID = [103818
105923
111312
114823
115320
122317
125525
130518
135528
137128
139839
143325
144226
146129
149337
149741
151526
158035
169343
172332
175439
177746
185442
187547
192439
194140
195041
200109
200614
204521
250427
287248
433839
562345
599671
601127
660951
662551
783462
859671
861456
877168
917255];

% calculate glasser netmats
nets_RT = zeros(360,360,43);
for s = 1:43
    fprintf('nets subject %d\n',s);
    D = ft_read_cifti(sprintf('MMP_netmats/retest/%d_REST1_LR.ptseries.nii',ID(s)));
    N1 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/retest/%d_REST1_RL.ptseries.nii',ID(s)));
    N2 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/retest/%d_REST2_LR.ptseries.nii',ID(s)));
    N3 = corr(D.ptseries'); clear D
    D = ft_read_cifti(sprintf('MMP_netmats/retest/%d_REST2_RL.ptseries.nii',ID(s)));
    N4 = corr(D.ptseries'); clear D
    nets_RT(:,:,s) = mean(cat(3,N1,N2,N3,N4),3); clear N1 N2 N3 N4 D
end
save('nets_retest.mat','nets_RT');

