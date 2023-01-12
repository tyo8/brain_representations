% computes a list of admissible permutations from a file defining a set of Exchnageability Blocks [1]
% [1] Winkler et al. (2015) https://doi.org/10.1016/j.neuroimage.2015.05.092
% requires Matlab library PALM

function [] = quickperm_fn(EB_fileloc,n_perm,floc_out)

addpath(genpath("PALM/"))

EB = readmatrix(EB_fileloc);

permset = palm_quickperms([],EB,n_perm+1);
permset(:,1) = [];

writematrix(permset,floc_out)

end
