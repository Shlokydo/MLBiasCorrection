#!/bin/bash

loc="../DATA/16_256_8_30000_en_20_obe_0.1_infl_2.8/test_obs/test_da/assim.nc"
ops="tt"
echo $loc
python Exp.py --t optimize -e 15 -nt 1 -l 9 -ts 3 -ncdf_loc $loc -tb 65536 -nbs 2 -os $ops -mr False


#usage: Exp.py [-h] [--t {optimize,best}] [--epochs EPOCHS]
#              [--num_trials NUM_TRIALS] [--netcdf_dataset NETCDF_DATASET]
#              [--locality LOCALITY] [--time_splits TIME_SPLITS]
#
#Optuna xperiment Controller
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
