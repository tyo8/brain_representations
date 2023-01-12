
run /vols/Data/HCP/Phase2/scripts1200/setup.m
M='ts2';
NetmatMethod=2;

d=300;  % the group-ICA dimensionality you want to process

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

F=sprintf('%s_d%d',BATCH,d);  G=sprintf('%s_%s',F,M);  NM=sprintf('%s/netmats/%s',SCRATCH,G);
SUMPICS=sprintf('%s/groupICA/groupICA_%s.ica/melodic_IC_sum',SCRATCH,F);
mkdir(NM);

ts_dir=sprintf('%s/node_timeseries/%s',SCRATCH,G);
ts=nets_load(ts_dir,0.72,1,4);

% estimate basic ts stats and spectra
[ts_stats,ts_all_stats] = nets_stats(ts);
ts_spectra=nets_spectra(ts);
%set(gcf,'PaperPositionMode','auto','Position',[10 10 2000 1300]);print('-dpng',sprintf('%s/ts_spectra.png',NM));close;

% estimate netmats at subject and group level
netmat1=nets_netmats(ts,1,'corr');
netmat2=nets_netmats(ts,1,'ridgep',0.01); % small amount of L2 regularisation for the partial netmats
grot=[];  for i=1:ts.Nsubjects/4, grot=[grot; mean(netmat1((i-1)*4+1:i*4,:))]; end; netmat1=2*grot;
grot=[];  for i=1:ts.Nsubjects/4, grot=[grot; mean(netmat2((i-1)*4+1:i*4,:))]; end; netmat2=2*grot;
[Znet1,Mnet1]=nets_groupmean(netmat1,0,1);
[Znet2,Mnet2]=nets_groupmean(netmat2,0,1);

% save matlab workspace with and without full node timeseries saved
save(sprintf('%s/workspace_with_ts',NM),'-v7.3');
ts=rmfield(ts,'ts');
save(sprintf('%s/workspace',NM),'-v7.3');

% hierarchical clustering of nodes
set(0,'DefaultFigureVisible','off');
[hier,linkages]=nets_hierarchy(Mnet1,4*Mnet2,ts.DD,SUMPICS);
set(gcf,'PaperPositionMode','auto','Position',[1 1 46*(ts.Nnodes+1) 1574 ]); % print('-dpng',sprintf('%s/hierarchy.png',NM)); 
close; set(0,'DefaultFigureVisible','on');

% save out group netmats in workbench format
    % make dummy NODETIMESERIES.ptseries.nii
system(sprintf('cp %s/100206/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii %s/tmp.dtseries.nii',SUBJECTS,FMRIB));
system(sprintf('%s -cifti-parcellate %s/tmp.dtseries.nii %s/groupICA/groupICA_%s_d%d.ica/melodic_IC_ftb.dlabel.nii COLUMN %s/tmp_d%d.ptseries.nii', ...
       WBC,FMRIB,SCRATCH,BATCH,d,FMRIB,d));
    % if we want a real subjects' .ptseries.nii, do:
    % grot=ciftiopen(sprintf('%s/tmp_d%d.ptseries.nii',FMRIB,d),WBC);
    % overwrite .data contents in last file with actual node timeseries and re-save
    % now make dummy NETMAT.pconn.nii
system(sprintf('%s -cifti-correlation %s/tmp_d%d.ptseries.nii %s/tmp_d%d.pconn.nii',WBC,FMRIB,d,FMRIB,d));
    % now can overwrite .data with actual netmats and re-save
grot=ciftiopen(sprintf('%s/tmp_d%d.pconn.nii',FMRIB,d),WBC);
grot.cdata=Mnet1;  ciftisave(grot,sprintf('%s/Mnet1.pconn.nii',NM),WBC);
grot.cdata=Mnet2;  ciftisave(grot,sprintf('%s/Mnet2.pconn.nii',NM),WBC);

% save out individual subject netmats
dlmwrite(sprintf('%s/netmats1.txt',NM),netmat1,'delimiter',' ');
dlmwrite(sprintf('%s/netmats2.txt',NM),netmat2,'delimiter',' ');
% save out individual subject netmats as pconns
%allsub=load(sprintf('%s/subjectIDs.txt',SCRATCH));
%mkdir(sprintf('%s_netmat1',NM));  mkdir(sprintf('%s_netmat2',NM));
%for i=1:length(allsub)
%  grot.cdata=reshape(netmat1(i,:),ts.Nnodes,ts.Nnodes);   ciftisave(grot,sprintf('%s_netmat1/%d.pconn.nii',NM,allsub),WBC);
%  grot.cdata=reshape(netmat2(i,:),ts.Nnodes,ts.Nnodes);   ciftisave(grot,sprintf('%s_netmat2/%d.pconn.nii',NM,allsub),WBC);
%end

