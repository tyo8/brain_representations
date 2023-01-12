#!/bin/sh

. `dirname $0`/setup.sh

cd $SUBJECTS
mkdir -p ../subjects1200incomplete

for s in `echo [1-9]?????` ; do
  ss=$SUBJECTS/$s/MNINonLinear/Results
  a=$ss/rfMRI_REST1_LR/rfMRI_REST1_LR_
  b=$ss/rfMRI_REST1_RL/rfMRI_REST1_RL_
  c=$ss/rfMRI_REST2_LR/rfMRI_REST2_LR_
  d=$ss/rfMRI_REST2_RL/rfMRI_REST2_RL_
  e=hp2000_clean.nii.gz
  f=Atlas_MSMAll_hp2000_clean.dtseries.nii
  if [ -f ${a}$e -a -f ${b}$e -a -f ${c}$e -a -f ${d}$e -a -f ${a}$f -a -f ${b}$f -a -f ${c}$f -a -f ${d}$f ] ; then
    a=`fslval ${a}$e dim4`
    b=`fslval ${b}$e dim4`
    c=`fslval ${c}$e dim4`
    d=`fslval ${d}$e dim4`
    abcd=`echo "$a $b + $c + $d + p" | dc -`
    #echo $s $a $b $c $d $abcd
    if [ $abcd = 4800 ] ; then
      echo subject $s is good > /dev/null
    else
      echo subject $s does not have the full set of 4800 timepoints \(only has ${abcd}\)
      #mv $s ../subjects1200incomplete
    fi
  else
    echo subject $s is missing at least one fix-cleaned NIFTI or CIFTI file
    #mv $s ../subjects1200incomplete
  fi
done

for f in `echo [1-9]?????` ; do
  echo $f >> $SCRATCH/subjectIDs.txt
done

