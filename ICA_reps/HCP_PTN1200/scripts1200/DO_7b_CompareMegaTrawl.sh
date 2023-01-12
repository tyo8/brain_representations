#!/bin/sh

#/bin/rm /tmp/megatrawl_results.txt

for d in 200 ; do
  #for D in /vols/Data/HCP/Phase2/group1200/netmats/3T_HCP1200_MSMAll_d${d}_ts2 /vols/Data/HCP/Phase2/group1200_recon2/netmats/3T_HCP1200_MSMAll_d${d}_ts2 ; do
  for D in /vols/Data/HCP/Phase2/group1200_recon2/netmats/3T_HCP1200_MSMAll_d${d}_ts2  ; do
  cd $D
    for sm in megatrawl_1/sm? megatrawl_1/sm?? megatrawl_1/sm??? ; do
      if [ `cat $sm/index.html | grep 'Either not enough subjects' | wc -l` != 1 ] ; then
        r1=`grep "Original data space" $sm/index.html | awk -F = '{print $2}' | awk '{print $1}'`
        c1=`grep "Original data space" $sm/index.html | awk -F = '{print $3}' | awk '{print $1}'`
        r2=`grep "Original data space" $sm/index.html | awk -F = '{print $4}' | awk '{print $1}'`
        c2=`grep "Original data space" $sm/index.html | awk -F = '{print $5}' | awk '{print $1}'`
        echo $r1 $r2 $c1 $c2 >> /tmp/megatrawl_results.txt
      else
        echo "NaN NaN NaN NaN" >> /tmp/megatrawl_results.txt
      fi
    done
  done
done

exit
% matlab stuff:

grotN=4;
grot=load('/tmp/megatrawl_results.txt');  grot=reshape(grot,size(grot,1)/grotN,[]);

for I=1:4
  poop=grot(:,(I-1)*grotN+[1:grotN]); poopy=mean(poop,2); subplot(2,2,I);
  for II=1:grotN
    scatter(poopy,poop(:,II),'.'); hold on;
  end
end

legend('d25 all','d25 recon2','d200 all','d200 recon2');


