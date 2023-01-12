#!/bin/sh

. `dirname $0`/setup.sh

NodeMethod=2

for d in 15 25 50 100 200 300 ; do

  cd $SCRATCH/node_maps/${BATCH}_d${d}_ts${NodeMethod}_Z
  N=`echo *.nii.gz | wc -w`
  echo found $N NIFTI files
  G=$SCRATCH/groupICA/groupICA_${BATCH}_d${d}.ica
  fsladd $G/melodic_IC_sum *.nii.gz
  cd $G

  ROOTN=`echo "1 k $N v p" | dc -`
  fslmaths melodic_IC_sum -div $ROOTN melodic_IC_sum
  fslmaths melodic_IC_sum -Tmax -thr 15 -bin melodic_IC_sum_maxn
  fslmaths melodic_IC_sum -Tmaxn -add 1 -mas melodic_IC_sum_maxn melodic_IC_sum_maxn

  fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain -thr 4500 -bin -dilF -ero -fillh bg
  fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain -add 1 -mas bg bg
  slices_summary melodic_IC_sum 50 bg melodic_IC_sum.sum -1 -d
  cd melodic_IC_sum.sum;     for f in ????.png ; do pngtopnm $f | pnmtopng -transparent=black > grot.png; /bin/mv grot.png $f; done

done

