
run /vols/Data/HCP/Phase2/scripts1200/setup.m

% setup brainordinates struct and get number of brainordinates
BO=ciftiopen(sprintf('%s/migp/groupPCA_%s_d4500.dtseries.nii',SCRATCH,BATCH),WBC);  BOdimX=size(BO.cdata,1);  BOdimZnew=ceil(BOdimX/100); clear BO.cdata;

for d = [ 10 12 15 20 21 22 23 24 25 50 100 200 300 ]
  G=sprintf('groupICA_%s_d%d.ica',BATCH,d)
  cd(sprintf('%s/groupICA/%s',SCRATCH,G));

  % read NIFTI-format brainordinate ICA spatial maps and resave as CIFTI
  GM=read_avw(sprintf('melodic_IC'));  gt=size(GM,4); % need gt as might not have got all the components we asked for
  GM=reshape(GM,100*BOdimZnew,gt);  GM=GM(1:BOdimX,:);  

  % make all individual group maps have a positive peak, and of peak height=1
  BO.cdata=GM.*repmat(sign(max(GM)+min(GM)),size(GM,1),1)./repmat(max(abs(GM)),size(GM,1),1);  
  ciftisave(BO,sprintf('melodic_IC.dtseries.nii'),WBC);

  % save "parcellation" as index number of max (across components) spatial map value (not using max=1 normalisation)
  BO.cdata=GM.*repmat(sign(max(GM)+min(GM)),size(GM,1),1);  [yy,ii]=sort(BO.cdata,2);  BO.cdata=[ii(:,end)];
  ciftisave(BO,sprintf('melodic_IC_maxn.dtseries.nii'),WBC);

  % create dscalar and other versions of the CIFTI - NFI why
  fidb=fopen('.list','w');
  for dd=1:d
    fprintf(fidb,'%d\n',dd);
  end
  fclose(fidb);
  ! wb_command -cifti-convert-to-scalar melodic_IC.dtseries.nii ROW melodic_IC.dscalar.nii -name-file .list
  ! wb_command -cifti-reduce melodic_IC.dscalar.nii INDEXMAX melodic_IC_ftb.dscalar.nii
  ! wb_command -cifti-label-import melodic_IC_ftb.dscalar.nii "" melodic_IC_ftb.dlabel.nii

end

