#!/bin/sh


for i in `seq 50`;do
  echo $i" ..."
  inittime=`expr $i \* 500` 
  python forecast.py $inittime
done
