#!/bin/sh

. $(dirname $0)/setup.sh

cd $SUBJECTS

for s in `echo ??????` ; do
  echo $s
  cd $s
  echo "subID=$s; DIMS=[15 25 50 100 200 300]; NodeTimeseriesMethod=2; run ${FMRIB}/setup.m; subproc_DO_4_NodeTS" > grot.m
  fsl_sub -q maf.q /opt/fmrib/MATLAB/R2014a/bin/matlab -nojvm -nodisplay -nosplash \< grot.m

  cd ..
done

