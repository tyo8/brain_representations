#!/bin/sh

i=1
Nsubsets=32

while [ $i -le $Nsubsets ] ; do
  echo $i
  echo "iam=$i; Nsubsets=$Nsubsets; subproc_DO_2_groupPCA" > grot_${i}.m
  fsl_sub -q maf.q /opt/fmrib/MATLAB/R2014a/bin/matlab -nojvm -nodisplay -nosplash \< grot_${i}.m
  sleep 10
  i=`echo $i 1 + p | dc -`
done

exit

calculation: 10 mins per MIGP iteration
T1 = 6 * 4000 / N   (in minutes)
T2 = 60 * N  (assuming multithreaded)
T = T1 + T2 = 24000 / N + 60 * N
dT/dN = -24000/N^2 + 60 = 0
N^2 = 400
N=20
T = 2days

g=50 (GB per MAF job)
lx64_20HT  1:7,20      300G     N = 8*300/g
lx64_28HT  21:23       500G     N = 3*500/g
lx64_32    12:17,27:28 500G     N = 8*500/g

