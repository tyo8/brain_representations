
addpath(sprintf('%s/etc/matlab',getenv('FSLDIR')))

ff{1}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,subID);
ff{2}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,subID);
ff{3}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,subID);
ff{4}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS,subID);

gg{1}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii',SUBJECTS,subID);
gg{2}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii',SUBJECTS,subID);
gg{3}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_hp2000_clean.nii',SUBJECTS,subID);
gg{4}=sprintf('%s/%d/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_hp2000_clean.nii',SUBJECTS,subID);

for i=1:4
  i
  BO=ciftiopen(ff{i},WBC);  SUB{i}=BO.cdata; clear BO;
  SUBv{i}=read_avw(gg{i});
  if i==1
    SUBvX=size(SUBv{1},1); SUBvY=size(SUBv{1},2); SUBvZ=size(SUBv{1},3); SUBvT=size(SUBv{1},4);
  end
  SUBv{i}=nets_demean(reshape(SUBv{i},SUBvX*SUBvY*SUBvZ,SUBvT)')';
end

for d=DIMS
  d
  system(sprintf('mkdir -p %s/node_timeseries/%s_d%d_ts%d',SCRATCH,BATCH,d,NodeTimeseriesMethod));
  system(sprintf('mkdir -p %s/node_maps/%s_d%d_ts%d_Z',SCRATCH,BATCH,d,NodeTimeseriesMethod));

  BO=ciftiopen(sprintf('%s/groupICA/groupICA_%s_d%d.ica/melodic_IC.dtseries.nii',SCRATCH,BATCH,d),WBC);
  pGM=pinv(nets_demean(BO.cdata));  allNODEts=[]; allMAPS=zeros(size(BO.cdata)); clear BO.cdata;

  if NodeTimeseriesMethod==4 % Matt Glasser's weighted-regression - load variability maps - placeholder code
    BO=ciftiopen(sprintf('%s/groupICA/groupICA_%s_d%d.ica/melodic_IC.dtseries.nii',SCRATCH,BATCH,d),WBC);
    pGM=pinv(nets_demean(BO.cdata));  allNODEts=[]; allMAPS=zeros(size(BO.cdata)); clear BO.cdata;
  end

  for i=1:4
    i
    if NodeTimeseriesMethod==2
      NODEts=nets_demean((pGM*nets_demean(SUB{i}))');
    end

    if NodeTimeseriesMethod==4 % Matt Glasser's weighted-regression
      NODEts=nets_demean((pGM*nets_demean(SUB{i}))');  % place-holder code - need to implement weighted-regression
    end

    allNODEts=[allNODEts ; NODEts]; NODEts=nets_normalise(NODEts); grot=nets_demean(SUB{i}');  TSd{i}=NODEts;   %%pTSd{i}=pinv(NODEts)'; 
    allMAPS = allMAPS + ssglmT(grot,NODEts)';
  end

  clear pGM;
  dlmwrite(sprintf('%s/node_timeseries/%s_d%d_ts%d/%d.txt',SCRATCH,BATCH,d,NodeTimeseriesMethod,subID),allNODEts,'delimiter',' '); clear allNODEts;
  BO.cdata=allMAPS/2; ciftisave(BO,sprintf('%s/node_maps/%s_d%d_ts%d_Z/%d.dtseries.nii',SCRATCH,BATCH,d,NodeTimeseriesMethod,subID),WBC); clear BO allMAPS;

  allMAPSv=zeros(SUBvX,SUBvY,SUBvZ,d);
  for i=1:4
    i
    %%allMAPSv = allMAPSv + reshape( SUBv{i} * pTSd{i} ,SUBvX,SUBvY,SUBvZ,d);
    allMAPSv = allMAPSv + reshape( ssglmT(SUBv{i}',TSd{i})' ,SUBvX,SUBvY,SUBvZ,d);
  end

  allMAPSv=allMAPSv/2; allMAPSv(isnan(allMAPSv))=0;
  save_avw(allMAPSv,sprintf('%s/node_maps/%s_d%d_ts%d_Z/%d',SCRATCH,BATCH,d,NodeTimeseriesMethod,subID),'f',[2 2 2 1]);
  clear allMAPSv pTSd;
  system(sprintf('fslcpgeom %s %s/node_maps/%s_d%d_ts%d_Z/%d -d',gg{1},SCRATCH,BATCH,d,NodeTimeseriesMethod,subID));

end

