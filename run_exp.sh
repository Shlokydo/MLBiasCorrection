#!/bin/bash

tdir=temp_test
cdir=$(pwd)
tfs=Python_files

mkdir -p $cdir/$tdir
cp $cdir/*.py $cdir/$tdir
cp -r $cdir/DATA $cdir/$tdir
cp -r $cdir/$tfs/* $cdir/$tdir
cd $cdir/$tdir
python exp.py
cp -rn $cdir/$tdir/DATA $cdir
rm -r $cdir/$tdir
