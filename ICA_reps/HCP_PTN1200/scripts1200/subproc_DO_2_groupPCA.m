
run /vols/Data/HCP/Phase2/scripts1200/setup.m

GO=sprintf('%s/migp',SCRATCH); mkdir(GO);
dPCAint=4700;        % just less than one subject of 4 runs  (used to be 4700)
allsub=dir(SUBJECTS);
Nsub=size(allsub,1)-2;
NinSubset=ceil(Nsub/Nsubsets);

ff=1
for s=(iam-1)*NinSubset+1:min(Nsub,iam*NinSubset)
  s
  f{ff}=sprintf('%s/%s/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,allsub(s+2).name); ff=ff+1;
  f{ff}=sprintf('%s/%s/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,allsub(s+2).name); ff=ff+1;
  f{ff}=sprintf('%s/%s/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,allsub(s+2).name); ff=ff+1;
  f{ff}=sprintf('%s/%s/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,allsub(s+2).name); ff=ff+1;
end

Nsub=ff-1; r=randperm(Nsub);

if Nsub>0

BO=ciftiopen(f{r(1)},WBC); grot=demean(double(BO.cdata)'); clear BO.cdata;
[uu,ss,vv]=ss_svds(grot,30); vv(abs(vv)<2.3*std(vv(:)))=0; stddevs=max(std(grot-uu*ss*vv'),0.001); grot=grot./repmat(stddevs,size(grot,1),1);  % var-norm
W=demean(grot); clear grot;

for i=2:Nsub
  i
  BO=ciftiopen(f{r(i)},WBC); grot=demean(double(BO.cdata)'); clear BO.cdata;
  if size(grot,1)<1200
    sprintf('ERROR INCOMPLETE DATA FOR %s',f{r(i)})
  else
    [uu,ss,vv]=ss_svds(grot,30); vv(abs(vv)<2.3*std(vv(:)))=0; stddevs=max(std(grot-uu*ss*vv'),0.001); grot=grot./repmat(stddevs,size(grot,1),1);  % var-norm
    W=[W; demean(grot)]; clear grot;
    if size(W,1)-10 > dPCAint
      [uu,dd]=eigs(W*W',dPCAint);  W=uu'*W; clear uu;
    end
  end
end

save(sprintf('%s/migp_%.3d.mat',GO,iam),'W','-v7.3');

end

