#!/bin/sh

cd /vols/Scratch/HCP/rfMRI

for f in subjects900/?????? ; do
  f=`basename $f`
  if [ -d subjects1200/$f ] ; then
    echo already have $f - doing nothing
  else
    mv subjects900/$f subjects1200
    cd subjects900
    ln -s ../subjects1200/$f .
    cd ..
  fi
done

