#!/bin/bash

if [ $1 = test ]
then
    python testing.py -en $2 -ts $3 -me $4
else
    #loc="/home/amemiya/python/MLBiasCorrection/DATA/coupled_A13/obs_010/nocorr/assim.nc"
    loc="/home/amemiya/python/MLBiasCorrection/DATA/coupled_A13/obs_p6_010/nocorr/assim.nc"
    for i in 1 3 5 
    do
      ops="coupled_A13_p6_LD_tanh10_$i"
      echo $ops
      python Exp.py --t $1 -e 30 -nt 30 -l $i -ts 10 -ncdf_loc $loc -tb 32768 -nbs 1 -os $ops -mr 1 -vb 2048 -norm 0 -d 1 -afm 0 -osql "coupled_A13"
      python Exp.py --t best -e 300 -nt 30 -l $i -ts 10 -ncdf_loc $loc -tb 32768 -nbs 1 -os $ops -mr 1 -vb 2048 -norm 0 -d 1 -afm 0 -osql "coupled_A13"
    done
fi

#usage: Exp.py [-h] [--t {optimize,best,p_best}] [--m_recu {0,1}]
#              [--epochs EPOCHS] [--num_trials NUM_TRIALS]
#              [--netcdf_dataset NETCDF_DATASET] [--optuna_study OPTUNA_STUDY]
#              [--optuna_sql OPTUNA_SQL] [--locality LOCALITY]
#              [--degree DEGREE] [--normalized {0,1}] [--af_mix {0,1}]
#              [--time_splits TIME_SPLITS] [--train_batch TRAIN_BATCH]
#              [--val_batch VAL_BATCH] [--num_batches NUM_BATCHES]
#
#Optuna Experiment Controller
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --t {optimize,best,p_best}
#                        Choose between optimization or training best parameter
#  --m_recu {0,1}, -mr {0,1}
#                        Use LSTM layers or not
#  --epochs EPOCHS, -e EPOCHS
#                        Num of epochs for training
#  --num_trials NUM_TRIALS, -nt NUM_TRIALS
#                        Num of Optuna trials
#  --netcdf_dataset NETCDF_DATASET, -ncdf_loc NETCDF_DATASET
#                        Location of the netCDF dataset
#  --optuna_study OPTUNA_STUDY, -os OPTUNA_STUDY
#                        Optuna Study name
#  --optuna_sql OPTUNA_SQL, -osql OPTUNA_SQL
#                        Optuna Study name
#  --locality LOCALITY, -l LOCALITY
#                        Locality size (including the main variable)
#  --degree DEGREE, -d DEGREE
#                        To make a polynomial input
#  --normalized {0,1}, -norm {0,1}
#                        Use normalized dataset for training.
#  --af_mix {0,1}, -afm {0,1}
#                        Use analysis forecast mixed.
#  --time_splits TIME_SPLITS, -ts TIME_SPLITS
#                        Num of RNN timesteps
#  --train_batch TRAIN_BATCH, -tb TRAIN_BATCH
#                        Training batch size
#  --val_batch VAL_BATCH, -vb VAL_BATCH
#                        Validation batch size
#  --num_batches NUM_BATCHES, -nbs NUM_BATCHES
#                        Number of training batch per epoch
