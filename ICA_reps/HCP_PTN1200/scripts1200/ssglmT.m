function [T] = ssglmT(Y,X)

pinvX=pinv(X);

if size(X,1)<100
  R=eye(size(X,1))-X*pinvX;
  trR=trace(R);
  nu=round(trace(R)^2/trace(R*R));
else
  nu=size(X,1)-size(X,2);
  trR=nu;
end

pe=pinvX*Y;
res=Y-X*pe;
sigsq=sum(res.^2)/trR;

varcope=diag(pinvX*pinvX')*sigsq;

T = pe ./ sqrt(varcope);

