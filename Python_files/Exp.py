import copy
import random
import numpy as np
import pandas as pd
import os
import sys
import shutil
import argparse

import helperfunctions as helpfunc
import training_test as tntt
import testing as tes
import tensorflow as tf

import optuna

study = optuna.create_study(direction = 'minimize', study_name = 'RNN_bc', pruner = optuna.pruners.PercentilePruner(80.0), storage = 'sqlite:///mshlok/rnn_bc.db', load_if_exists = True)

parser = argparse.ArgumentParser(description = 'Optuna Experiment Controller')

parser.add_argument("--t", default = "optimize", type = str, choices = ["optimize", "best"], help = "Choose between optimization or training on best parameters (as of now)")
parser.add_argument("--epochs", "-e",  default = 200, type = int, help = "Num of epochs for training")
parser.add_argument("--num_trials", "-nt",  default = 10, type = int, help = "Num of Optuna trials")
parser.add_argument("--netcdf_dataset", "-ncdf_loc", default = "../DATA/simple_test/test_obs/test_da/assim.nc", type = str, help = "Location of the netCDF dataset")
parser.add_argument("--locality", "-l",  default = 9, type = int, help = "Locality size (including the main variable)")
parser.add_argument("--time_splits", "-ts",  default = 5, type = int, help = "Num of RNN timesteps")
args = parser.parse_args()

def my_config(trial):
    #Parameter List
    plist = {}
    
    #Network related settings
    plist['make_recurrent'] = True 

    plist['num_lstm_layers'] = trial.suggest_int('lstm_layers', 1, 2)
    plist['LSTM_output'] = []
    for i in range(plist['num_lstm_layers']):
        plist['LSTM_output'].append(trial.suggest_int('lstm_' + str(i), 2, 10)) 

    plist['num_dense_layers'] = trial.suggest_int('dense_layers', 1, 3) 
    plist['dense_output'] = []
    for i in range(plist['num_dense_layers']):
        plist['dense_output'].append(trial.suggest_int('dense_' + str(i), 2, 7))
    plist['dense_output'].append(1)

    plist['activation'] = 'tanh'
    plist['rec_activation'] = 'sigmoid'
    plist['l2_regu'] = 0.0
    plist['l1_regu'] = 0.0
    plist['lstm_dropout'] = 0.0
    plist['rec_lstm_dropout'] = 0.0
    
    plist['learning_rate'] = trial.suggest_uniform('learning_rate', 5e-4, 5e-3)

    #Dataset and directories related settings
    plist['netCDf_loc'] = args.netcdf_dataset
    plist['xlocal'] = 3
    plist['locality'] = args.locality
    plist['num_timesteps'] = 2020
    plist['time_splits'] = args.time_splits
    plist['experiment_name'] = args.t + '_L' + '_'.join(map(str, plist['LSTM_output'])) + '_D' + '_'.join(map(str, plist['dense_output'])) + '_lr' + str(plist['learning_rate'])
    plist['experiment_dir'] = './n_experiments/' + plist['experiment_name'] 
    plist['checkpoint_dir'] = plist['experiment_dir'] + '/checkpoint'
    plist['log_dir'] = plist['experiment_dir'] + '/log'

    plist['pickle_name'] = plist['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(plist['experiment_dir']):
        os.makedirs(plist['log_dir'])
        os.mkdir(plist['checkpoint_dir'])

        #Training related settings
        plist['max_checkpoint_keep'] = 3
        plist['log_freq'] = 1
        plist['early_stop_patience'] = 500
        plist['summery_freq'] = 1
        plist['global_epoch'] = 0
        plist['global_batch_size'] = 2048 
        plist['val_size'] = 1 * plist['global_batch_size']
        plist['lr_decay_steps'] = 300000
        plist['lr_decay_rate'] = 0.10
        try:
            plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128 * len(tf.config.experimental.list_physical_devices('GPU')))
        except:
            plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128)
        plist['val_min'] = 1000

    else:
        if os.path.isfile(plist['pickle_name']):
            plist = helpfunc.read_pickle(plist['pickle_name'])
        else:
            print('\nNo pickle file exists at {}. Exiting....\n'.format(plist['pickle_name']))
            shutil.rmtree(plist['experiment_dir'])
            sys.exit()

    plist['epochs'] = args.epochs
    plist['test_num_timesteps'] = 300
    
    return plist

def objective(trial):

    plist = my_config(trial)
    return tntt.traintest(trial, copy.deepcopy(plist))
    
if __name__ == "__main__":

    if args.t == 'optimize':
        study.optimize(objective, n_trials = args.num_trials)
        df = study.trials_dataframe()
        df.to_csv('optuna_exp.csv')
    elif args.t == 'best':
        best_case = optuna.trial.FixedTrial(study.best_params)
        objective(best_case)
