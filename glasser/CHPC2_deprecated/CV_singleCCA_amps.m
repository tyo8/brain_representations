clear all; close all; clc

% Set parameters
K = 10;
Num_features = 50;
Nsubs = 1000;

% Load data
NIHaffect = load('NIHaffect.csv');
ID = load('HCP_IDs.csv');
load('amps.mat'); amps = amps';

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
cca_out = zeros(4,K);
UVtest = zeros(Nsubs,2);
for k = 1:K
    fprintf('----------- CV FOLD %d -----------\n',k)
    test = find(Fold==k);
    train = setdiff(1:Nsubs,test);
    r = corr(NIHaffect(train,18),amps(train,:));
    [r,i] = sort(abs(r),'descend'); 
    features(:,k) = i(1:Num_features); clear r i
    
    [p,r,A,B,~,~,~,~] = permcca(amps(train,features(:,k)),NIHaffect(train,1:6),100);
    cca_out(1,k) = r(1); cca_out(2,k) = p(1); clear p r
    Utest = amps(test,features(:,k))*A;
    Vtest = NIHaffect(test,1:6)*B;
    [r,p] = corr(Utest,Vtest);
    cca_out(3,k) = r(1); cca_out(4,k) = p(1);
    UVtest(test,1) = Utest(:,1); UVtest(test,2) = Vtest(:,1);
    clear p r A B
end
r = corr(UVtest(:,1),UVtest(:,2));
t = r*sqrt((Nsubs-2)/(1-r^2));
p = 1-tcdf(t,Nsubs-1);
[mean(cca_out') r p]
