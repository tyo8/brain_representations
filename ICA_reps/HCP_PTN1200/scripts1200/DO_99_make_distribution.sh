#!/bin/sh

. `dirname $0`/setup.sh

cd $SCRATCH

D=$SCRATCH/HCP_PTN1200
mkdir -p $D
cd $D/..

cp subjectIDs*.txt $D
# subjectIDs.txt   Ordered list of the subjects included in this PTN release (and separate lists for recon1-only and recon2-only sets of subjects)

tar cvfz $D/groupICA_3T_HCP1200_MSMAll.tar.gz groupICA/groupICA_3T_HCP1200_MSMAll_d{15,25,50,100,200,300}.ica/{melodic_IC.dscalar.nii,melodic_IC_ftb.dlabel.nii,melodic_IC_sum.nii.gz,melodic_IC_sum.sum}
# groupICA_3T_HCP1200_MSMAll.tar.gz
# Group-ICA "parcellations" at several dimensionalities (levels of detail)
#    melodic_IC.dscalar.nii     ICA spatial maps (unthresholded Zstats); one "timepoint" per map.                     Grayordinates.
#    melodic_IC_ftb.dlabel.nii  Summary "find the biggest" labels image for all ICA spatial maps.                     Grayordinates.
#    melodic_IC_sum.nii.gz      ICA maps dual-regressed into subjects' 3D data and then averaged across subjects.     MNI152 space.
#    melodic_IC_sum.sum         Summary "thumbnail" PNG images created at the most relevant axial slices(s).          Slices of MNI152 space.
# 

for d in 15 25 50 100 200 300 ; do
  for t in ts2 ; do
    tar cvfz $D2/NodeTimeseries_3T_HCP1200_MSMAll_ICAd${d}_${t}.tar.gz node_timeseries/3T_HCP1200_MSMAll_d${d}_$t
    tar cvfz $D2/netmats_3T_HCP1200_MSMAll_ICAd${d}_${t}.tar.gz netmats/3T_HCP1200_MSMAll_d${d}_${t}/{netmats?.txt,Mnet?.pconn.nii}
  done
done
# NodeTimeseries_3T_HCP1200_MSMAll_ICAd*_ts*.tar.gz
# Node-timeseries, with one tarfile for each choice of group-ICA dimensionality and timeseries estimation approach
#    Inside each tarfile there is one timeseries text file per subject (concatenated across all 4 runs)
#    Within each text file there is one column per "node" (ICA component)
#    ICAd25 (etc) describes the original group-ICA dimensionality
#    ts2: multiple regression (against the set of ICA spatial maps) is used to estimate node timeseries (same as first stage of "dual regression")
# 
# netmats_3T_HCP1200_MSMAll_ICAd*_ts*.tar.gz
# Netmats (parcellated connectomes), with one tarfile for each choice of group-ICA dimensionality
#    netmats1.txt     All subjects' netmats: one subject's unwrapped netmat per row, computed using full correlation, Z-transformed.
#    netmats2.txt     As above, but using partial correlation with modest Tikhonov regularisation ("ridgep" in FSLNets with parameter 0.01)
#    Mnet1.pconn.nii  Group-average full correlation netmat
#    Mnet2.pconn.nii  As above, but for partial correlation
#    


# make MegaTrawl web outputs
#cd /vols/Data/HCP/Phase2/groupE/netmats
#tar cvfz grot.tar.gz *ts?d/{heritability/{aggher.png,agghernode.png,edgeher.png,index.html},hierarchy.png,index.html,netjs,netpics,papaya,sm*/{*.png,index.html}}

cd ..
tar cvfz $D/scripts.tar.gz scripts1200/{*.sh,*.m,README}

cd $SCRATCH/node_maps
tar cvfz ../HCP1200_PTNmaps_d15_25_50_100.tar.gz 3T_HCP1200_MSMAll_d{15,25,50,100}_ts2_Z/*dtseries.nii
tar cvfz ../HCP1200_PTNmaps_d200.tar.gz 3T_HCP1200_MSMAll_d200_ts2_Z/*dtseries.nii
tar cvfz ../HCP1200_PTNmaps_d300.tar.gz 3T_HCP1200_MSMAll_d300_ts2_Z/*dtseries.nii

