clear all; close all; clc

% Set parameters
K = 10;
Num_features = 50;
Nsubs = 1000;

% Load data
NIHaffect = load('NIHaffect.csv');
ID = load('HCP_IDs.csv');
load('nets.mat'); nets_new = zeros(Nsubs,360*359/2); I = ones(360); I = triu(I,1); I = find(I);
for s = 1:Nsubs
    N = squeeze(nets(:,:,s));
    nets_new(s,:) = N(I);
end

% Leave families out
Family = load('Family.csv');
F = unique(Family); % length 427 = 7*43 + 3*42 families 
F = F(randperm(length(F)));
F1 = F(1:7*43); F2 = F(7*43+1:end);
Fold = zeros(Nsubs,1); Fold_size = zeros(10,1);
for n = 1:7
    fam = F1((n-1)*43+1:n*43);
    for f = fam'
        Fold(Family==f) = n;
    end
    Fold_size(n) = length(find(Fold==n));
end
for n = 1:3
    fam = F2((n-1)*42+1:n*42);
    for f = fam'
        Fold(Family==f) = n+7;
    end
    Fold_size(n+7) = length(find(Fold==n+7));
end
Fold_size

% Set paths
addpath(genpath('/home/janine/Matlab/PermCCA/'));

% Perform feature selection + CCA in 10-fold cross-validation loop
features = zeros(Num_features,K);
cca_out = zeros(4,K); cca_out2 = zeros(4,K);
for k = 1:K
    fprintf('----------- CV FOLD %d -----------\n',k)
    test = find(Fold==k);
    train = setdiff(1:Nsubs,test);
    
    % Feature selection 
    r = corr(NIHaffect(train,18),nets_new(train,:));
    [r,i] = sort(abs(r),'descend'); 
    features(:,k) = i(1:Num_features); clear r i
    [p,r,A,B,~,~,~,~] = permcca(nets_new(train,features(:,k)),NIHaffect(train,1:6),100);
    cca_out(1,k) = r(1); cca_out(2,k) = p(1); clear p r
    Utest = nets_new(test,features(:,k))*A;
    Vtest = NIHaffect(test,1:6)*B;
    [r,p] = corr(Utest,Vtest);
    cca_out(3,k) = r(1); cca_out(4,k) = p(1);
    clear p r A B
    
    % PCA
    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(nets_new(train,:),'NumComponents',Num_features);
    fprintf('Variance explained by first PCA: %2.1f\n',EXPLAINED(1));
    [p,r,A,B,~,~,~,~] = permcca(SCORE,NIHaffect(train,1:6),100);
    cca_out2(1,k) = r(1); cca_out2(2,k) = p(1); clear p
    Utest = (nets_new(test,:)*COEFF)*A;
    Vtest = NIHaffect(test,1:6)*B;
    [r,p] = corr(Utest,Vtest);
    cca_out2(3,k) = r(1); cca_out2(4,k) = p(1);
    clear p r A B
end
mean(cca_out')
mean(cca_out2')
