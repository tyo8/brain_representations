
run /vols/Data/HCP/Phase2/scripts1200/setup.m

GO=sprintf('%s/migp',SCRATCH); cd(GO);
dPCAint=4700;        % just less than one subject of 4 runs

grot=load('migp_001.mat');
W=grot.W;

for i=2:1000
  grot=sprintf('migp_%.3d.mat',i);
  if  exist(grot) == 2
    grot
    grot=load(grot);
    W=[W; grot.W];
    [uu,dd]=eigs(W*W',dPCAint);  W=uu'*W; clear uu;
  else
    i=100000;
  end
end

save('migp_ALL.mat','W','-v7.3');

dPCA=4500;
BO=ciftiopen(sprintf('%s/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean.dtseries.nii',SUBJECTS),WBC);
BO.cdata=W(1:dPCA,:)';
ciftisave(BO,sprintf('groupPCA_%s_d%d.dtseries.nii',BATCH,dPCA),WBC);
TR=0.72; BOdimX=size(BO.cdata,1);  BOdimZnew=ceil(BOdimX/100);  BOdimT=size(BO.cdata,2);
save_avw(reshape([BO.cdata ; zeros(100*BOdimZnew-BOdimX,BOdimT)],10,10,BOdimZnew,BOdimT),sprintf('groupPCA_%s_d%d',BATCH,dPCA),'f',[1 1 1 TR]);

%%%% dumb dense connectome (RoF and noisier than the "proper" version below)
system(sprintf('wb_command -cifti-correlation groupPCA_%s_d%d.dtseries.nii groupPCA_%s_d%d_RawDenseConnectome.dconn.nii -fisher-z',BATCH,dPCA,BATCH,dPCA));
BOdconn=ciftiopen(sprintf('groupPCA_%s_d%d_RawDenseConnectome.dconn.nii',BATCH,dPCA),WBC); % read in to get the dconn structure into matlab

%%%% MGTR doubly-dumb dense connectome
%W=demean(W(1:dPCA,:));
%Wm=mean(W,2);
%W = W - Wm * (pinv(Wm)*W);
%BO.cdata=W';  ciftisave(BO,sprintf('groupPCA_%s_d%d_MGTR.dtseries.nii',BATCH,dPCA),WBC);
%system(sprintf('wb_command -cifti-correlation groupPCA_%s_d%d_MGTR.dtseries.nii groupPCA_%s_d%d_MGTR.dconn.nii -fisher-z',BATCH,dPCA,BATCH,dPCA));

%%%% MIGP-based dense connectome - with Wishart-rolloff to improve CNR and avoid Ring-of-Fire
MIGP=BO;
 %i=4500; id=400; grot=[1:i]'; grot=(0.2+0.8*exp(-(grot/id).^2)).*(1-(0.5*grot/i)).^4;   figure; plot(grot);
EigS=sqrt(sum(MIGP.cdata.^2))';       % get sqrt(eigenvalues)
EigD=EigS.^2; EigD=EigD/max(EigD);    %	get normalised eigenvalues
EigDn1=round(length(EigD)*0.6);
EigDn2=round(length(EigD)*0.8);
g1=10000; g3=100000;                  % golden search for best-fitting spatial-DoF
for i=1:20
  g2=(g1+g3)/2;
  EigNull=iFeta([0:0.0001:5],round(1.05*length(EigD)),g2)';
  EigNull=EigNull*sum(EigD(EigDn2:EigDn2+10))/sum(EigNull(EigDn2:EigDn2+10));
  grot=EigD(1:EigDn1)-EigNull(1:EigDn1);
  if min(grot)<0,  g1=g2;  else,  g3=g2;  end;
  [i g1 g2 g3]
end
EigDc=EigD-EigNull(1:length(EigD));  % subtract null eigenspectrum
grot=smooth(EigDc(50:end),50,'loess'); i=min(find(abs(grot)<1e-5))+50-10;
EigDc(i:end)= (1-(1:(1+length(EigDc)-i))/(1+length(EigDc)-i)).^2 * grot(i-50);
  subplot(1,3,1);   hold off; plot(EigNull  ); hold on; plot(EigD,'g'); plot(EigDc,'r'); hold off;
EigSc=sqrt(abs(EigDc)); EigScn=(EigSc>0).*EigSc./(EigS/EigS(1));  	% get correction factors
  subplot(1,3,2); plot([EigScn  EigSc]); % plot([EigScn  EigSc grot400 grot400.*EigS/EigS(1) grot4000 grot4000.*EigS/EigS(1)]);
  subplot(1,3,3); plot([ EigS EigS.*EigScn]);

% create Wishart-adjusted MIGP spatial eigenvectors
MIGProw = MIGP.cdata * diag(EigScn);
  BOdconn.cdata=MIGProw*MIGProw'; 
  BOdconn.cdata=(BOdconn.cdata ./ repmat(sqrt(abs(diag(BOdconn.cdata))),1,91282)) ./ repmat(sqrt(abs(diag(BOdconn.cdata)))',91282,1); 
  BOdconn.cdata=min(BOdconn.cdata,0.99999);
  BOdconn.cdata=0.5*log((1+BOdconn.cdata)./(1-BOdconn.cdata));
  ciftisave(BOdconn,sprintf('groupPCA_%s_d%d_novn2_DenseConnectome.dconn.nii',BATCH,dPCA),WBC);

% 2nd level varnorm of raw MIGP using sqrt(var(raw MIGP) - var(ROW MIGP))  then PCA
VN2=sqrt(var(MIGP.cdata,[],2) - var(MIGProw,[],2));  VN2=VN2/mean(VN2);
MIGPvn2 = MIGP.cdata ./ repmat(VN2,1,size(MIGProw,2));
[uu,ss,vv]=nets_svds(double(MIGPvn2),size(MIGPvn2,2)-1);

EigS=diag(ss);
EigD=EigS.^2; EigD=EigD/max(EigD);    %	get normalised eigenvalues
EigDn1=round(length(EigD)*0.6);
EigDn2=round(length(EigD)*0.8);
g1=10000; g3=100000;                  % golden search for best-fitting spatial-DoF
for i=1:20
  g2=(g1+g3)/2;
  EigNull=iFeta([0:0.0001:5],round(1.05*length(EigD)),g2)';
  EigNull=EigNull*sum(EigD(EigDn2:EigDn2+10))/sum(EigNull(EigDn2:EigDn2+10));
  grot=EigD(1:EigDn1)-EigNull(1:EigDn1);
  if min(grot)<0,  g1=g2;  else,  g3=g2;  end;
  [i g1 g2 g3]
end
EigDc=EigD-EigNull(1:length(EigD));  % subtract null eigenspectrum
grot=smooth(EigDc(50:end),50,'loess'); i=min(find(abs(grot)<1e-5))+50-10;
EigDc(i:end)= (1-(1:(1+length(EigDc)-i))/(1+length(EigDc)-i)).^2 * grot(i-50);
  figure; subplot(1,3,1);   hold off; plot(EigNull  ); hold on; plot(EigD,'g'); plot(EigDc,'r'); hold off;
EigSc=sqrt(abs(EigDc)); EigScn=(EigSc>0).*EigSc./(EigS/EigS(1));  	% get correction factors
  subplot(1,3,2); plot([EigScn  EigSc ]);
  subplot(1,3,3); plot([ EigS EigS.*EigScn]);

% create Wishart-adjusted MIGP spatial eigenvectors
MIGProw = uu * ss * diag(EigScn);
  BOdconn.cdata=MIGProw*MIGProw'; 
  BOdconn.cdata=(BOdconn.cdata ./ repmat(sqrt(abs(diag(BOdconn.cdata))),1,91282)) ./ repmat(sqrt(abs(diag(BOdconn.cdata)))',91282,1); 
  BOdconn.cdata=min(BOdconn.cdata,0.99999);
  BOdconn.cdata=0.5*log((1+BOdconn.cdata)./(1-BOdconn.cdata));
  ciftisave(BOdconn,sprintf('groupPCA_%s_d%d_DenseConnectome.dconn.nii',BATCH,dPCA),WBC);

