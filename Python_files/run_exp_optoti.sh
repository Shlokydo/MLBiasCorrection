#!/bin/bash

loc="../DATA/16_256_8_10000_en_20_obe_0.1_infl_2.8/test_obs/test_da/assim.nc"
ops="Trial_3"
echo $loc
python Exp.py --t optimize -e 500 -nt 10 -l 9 -ts 5 -ncdf_loc $loc -pvts 3940 -os $ops


#usage: Exp.py [-h] [--t {optimize,best}] [--epochs EPOCHS]
#              [--num_trials NUM_TRIALS] [--netcdf_dataset NETCDF_DATASET]
#              [--locality LOCALITY] [--time_splits TIME_SPLITS]
#
#Optuna Experiment Controller
#
#optional arguments:
#  -h, --help            show this help message and exit
#  --t {optimize,best}   Choose between optimization or training on best
#                        parameters (as of now)
#  --epochs EPOCHS, -e EPOCHS
#                        Num of epochs for training
#  --num_trials NUM_TRIALS, -nt NUM_TRIALS
#                        Num of Optuna trials
#  --netcdf_dataset NETCDF_DATASET, -ncdf_loc NETCDF_DATASET
#                        Location of the netCDF dataset
#  --locality LOCALITY, -l LOCALITY
#                        Locality size (including the main variable)
#  --time_splits TIME_SPLITS, -ts TIME_SPLITS
#                        Num of RNN timesteps
