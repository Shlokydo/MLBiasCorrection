#!/bin/sh
#set -e
VAR=_biased
METHOD=ML
#METHOD=reg
CONFIG=X08F08
OBS=all_01
#EXP=M20L30I20
#EXP=M20L30I20_A80NB99
#EXP=nocorr
EXP=${METHOD}
#EXP=DdSM
#EXP=reg
#EXP=test

F90="ifort -mkl -fpp"
DEBUG=
#DEBUG="-CU -CB -traceback"

CDIR=`pwd`
cd ..
L96DIR=`pwd`
cd ..
ENKFDIR=`pwd`
COMDIR=$ENKFDIR/common
OUTDIR=$L96DIR/DATA/$CONFIG
WKDIR=$L96DIR/tmp
BCDIR=$L96DIR/bc_offline
rm -rf $WKDIR
mkdir -p $WKDIR
cd $WKDIR
cp $COMDIR/SFMT.f90 .
cp $COMDIR/common.f90 .
cp $COMDIR/netlib.f .
cp $COMDIR/common_mtx.f90 .
cp $COMDIR/common_letkf.f90 .
cp $L96DIR/model/lorenz96$VAR.f90 .
cp $L96DIR/obs/h_ope.f90 .
cp $CDIR/letkf_offline_${METHOD}.f90 .
if [ "$METHOD" == "reg" ] ;then
cp $BCDIR/$METHOD/* .
$F90 -o letkf SFMT.f90 common.f90 netlib.f common_mtx.f90 common_letkf.f90 lorenz96$VAR.f90 h_ope.f90 interface_offline_${METHOD}.f90 letkf_offline_${METHOD}.f90 -lnetcdf -lnetcdff 
elif [ "$METHOD" == "ML" ] ;then
cp $BCDIR/$METHOD/*.py .
cp $BCDIR/$METHOD/*.F90 .
cp $BCDIR/$METHOD/*.f90 .
ln -s $BCDIR/$METHOD/n_experiments .
$F90 -o letkf SFMT.f90 common.f90 netlib.f common_mtx.f90 common_letkf.f90 lorenz96$VAR.f90 h_ope.f90 forpy_mod.F90 interface_offline_${METHOD}.f90 letkf_offline_${METHOD}.f90 -lnetcdf -lnetcdff `python3-config --ldflags` 

else
 echo 'unsupported method '$METHOD
 exit 1
fi


rm *.mod
rm *.o
ln -s $OUTDIR/$OBS/obs.nc .
ln -s $OUTDIR/spinup/init*.nc .
ln -s $OUTDIR/nature.nc .
time ./letkf
rm -rf $OUTDIR/$OBS/$EXP
mkdir -p $OUTDIR/$OBS/$EXP
mv assim.nc $OUTDIR/$OBS/$EXP
for FILE in infl rmse_t rmse_x
do
if test -f $FILE.dat
then
mv $FILE.dat $OUTDIR/$OBS/$EXP
fi
done


[ "$MONITOR" == "T" ] && mv monitor_obs_*.png $OUTDIR/$OBS/$EXP

echo "NORMAL END"
