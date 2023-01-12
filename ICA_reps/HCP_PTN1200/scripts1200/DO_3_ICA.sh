#!/bin/sh

. `dirname $0`/setup.sh

cd $SCRATCH/migp
mkdir -p ../groupICA

for d in 10 12 15 20 21 22 23 24 25 50 100 200 300 ; do

  fsl_sub -q bigmem.q melodic -i groupPCA_${BATCH}_d4500 -o ../groupICA/groupICA_${BATCH}_d${d}.ica -d $d --nobet --nomask --bgthreshold=-1e10 --Oall --report -v

done

