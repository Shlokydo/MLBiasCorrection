import copy
import random
import numpy as np
np.random.seed(1)
import pandas as pd
import os
import sys
import shutil
import argparse
import re

import helperfunctions as helpfunc
import training_test as tntt
import testing as tes
import tensorflow as tf
from tfdeterminism import patch
patch()

import optuna

parser = argparse.ArgumentParser(description = 'Optuna Experiment Controller')

parser.add_argument("--t", default = "optimize", type = str, choices = ["optimize", "best"], help = "Choose between optimization or training on best parameters (as of now)")
parser.add_argument("--epochs", "-e",  default = 200, type = int, help = "Num of epochs for training")
parser.add_argument("--num_trials", "-nt",  default = 10, type = int, help = "Num of Optuna trials")
parser.add_argument("--netcdf_dataset", "-ncdf_loc", default = "../DATA/simple_test/test_obs/test_da/assim.nc", type = str, help = "Location of the netCDF dataset")
parser.add_argument("--optuna_study", "-os", default = "simple_test", type = str, help = "Optuna Study name")
parser.add_argument("--locality", "-l",  default = 9, type = int, help = "Locality size (including the main variable)")
parser.add_argument("--time_splits", "-ts",  default = 5, type = int, help = "Num of RNN timesteps")
parser.add_argument("--train_batch", "-tb", default = 16384, type = int, help = "Training batch size")
parser.add_argument("--val_batch", "-vb", default = 16384, type = int, help = "Validation batch size")
parser.add_argument("--num_batches", "-nbs",  default = 1, type = int, help = "Number of training batch per epoch")
args = parser.parse_args()

def my_config(trial):
    #Parameter List
    plist = {}
    
    #Network related settings
    plist['make_recurrent'] = True 

    plist['num_lstm_layers'] = trial.suggest_int('lstm_layers', 1, 2)
    plist['LSTM_output'] = []
    plist['LSTM_output'].append(trial.suggest_int('lstm_' + str(0), 20, 40))
    for i in range(plist['num_lstm_layers'] - 1):
        plist['LSTM_output'].append(trial.suggest_int('lstm_' + str(i+1), 20, plist['LSTM_output'][i]))

    plist['num_dense_layers'] = trial.suggest_int('dense_layers', 3, 5) 
    plist['dense_output'] = []
    plist['dense_output'].append(trial.suggest_int('dense_' + str(0), 4, 20))
    for i in range(plist['num_dense_layers'] - 1):
        plist['dense_output'].append(trial.suggest_int('dense_' + str(i+1), 4, plist['dense_output'][i]))
    plist['dense_output'].append(1)

    plist['activation'] = 'tanh'
    plist['rec_activation'] = 'sigmoid'
    plist['l2_regu'] = 0.0
    plist['l1_regu'] = 0.0
    plist['lstm_dropout'] = 0.0
    plist['rec_lstm_dropout'] = 0.0

    #Training related settings
    plist['time_splits'] = args.time_splits
    plist['max_checkpoint_keep'] = 3
    plist['log_freq'] = 1
    plist['early_stop_patience'] = 50
    plist['summery_freq'] = 1
    plist['global_epoch'] = 0
    plist['global_batch_size'] = args.train_batch  
    plist['global_batch_size_v'] = args.val_batch
    plist['val_size'] = 1 * plist['global_batch_size_v']
    plist['num_timesteps'] = int(((plist['global_batch_size'] * args.num_batches + plist['val_size']) * plist['time_splits']/ 16) + 100)
    plist['lr_decay_steps'] = 1000
    plist['lr_decay_rate'] = trial.suggest_categorical('lrdr', [0.85, 0.95, 0.9])
    grad_mellow = trial.suggest_categorical('grad_mellow', [1, 0.1])
    plist['grad_mellow'] = grad_mellow
    plist['val_min'] = 1000
    plist['learning_rate'] = trial.suggest_loguniform('lr', 1e-3, 2e-3)
    try:
        plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128 * len(tf.config.experimental.list_physical_devices('GPU')))
    except:
        plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128)
    
    #Dataset and directories related settings
    plist['netCDf_loc'] = args.netcdf_dataset
    plist['xlocal'] = 3
    plist['locality'] = args.locality
    plist['experiment_name'] = args.t + '_L' + '_'.join(map(str, plist['LSTM_output'])) + '_D' + '_'.join(map(str, plist['dense_output'])) + '_lr' + str(plist['learning_rate'])[0:4]
    plist['experiment_dir'] = './n_experiments/' + plist['experiment_name'] 
    plist['checkpoint_dir'] = plist['experiment_dir'] + '/checkpoint'
    plist['log_dir'] = plist['experiment_dir'] + '/log'

    plist['pickle_name'] = plist['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(plist['experiment_dir']):
        os.makedirs(plist['log_dir'])
        os.mkdir(plist['checkpoint_dir'])
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
    
    user = os.getlogin()
    if user == 'mshlok':
        study = optuna.create_study(direction = 'minimize', study_name = args.optuna_study, pruner = optuna.pruners.PercentilePruner(80.0), storage = 'sqlite:///mshlok/' + str(re.search('/DATA/(.+?)/', args.netcdf_dataset).group(1)) + '.db', load_if_exists = True)
    elif user == 'amemiya':
        study = optuna.create_study(direction = 'minimize', study_name = args.optuna_study, pruner = optuna.pruners.PercentilePruner(80.0), storage = 'sqlite:///amemiya/' + str(re.search('/DATA/(.+?)/', args.netcdf_dataset).group(1)) + '.db', load_if_exists = True)

    if args.t == 'optimize':
        study.optimize(objective, n_trials = args.num_trials)
        df = study.trials_dataframe()
        df.to_csv('optuna_exp.csv')
    elif args.t == 'best':
        best_case = optuna.trial.FixedTrial(study.best_params)
        #best_case = optuna.trial.FixedTrial({'lstm_layers': 4, 'lstm_0': 128, 'lstm_1': 108, 'lstm_2': 107, 'lstm_3': 92, 'grad_mellow': 0.1, 'dense_layers': 1, 'dense_0': 20})
        objective(best_case)
