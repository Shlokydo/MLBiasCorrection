import copy
import random
import numpy as np
import pandas as pd
import os
import sys
import shutil

import helperfunctions as helpfunc
import training_test as tntt
import testing as tes
import tensorflow as tf

from sacred import Experiment
from sacred.observers import MongoObserver

import optuna

study = optuna.create_study(direction = 'minimize', study_name = 'RNN_bc', pruner = optuna.pruners.PercentilePruner(80.0))

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

    #Dataset and directories related settings
    plist['netCDf_loc'] = "../DATA/simple_test/test_obs/test_da/assim.nc"
    plist['xlocal'] = 3
    plist['locality'] = 9
    plist['num_timesteps'] = 2020
    plist['time_splits'] = 5
    plist['experiment_name'] = 'L' + '_'.join(map(str, plist['LSTM_output'])) + '_D' + '_'.join(map(str, plist['dense_output']))
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
        plist['lr_decay_rate'] = 0.70
        try:
            plist['learning_rate'] = 1e-3 * plist['global_batch_size'] / (128 * len(tf.config.experimental.list_physical_devices('GPU')))
        except:
            plist['learning_rate'] = 1e-3 * plist['global_batch_size'] / (128)
        plist['val_min'] = 1000

    else:
        if os.path.isfile(plist['pickle_name']):
            plist = helpfunc.read_pickle(plist['pickle_name'])
        else:
            print('\nNo pickle file exists at {}. Exiting....\n'.format(plist['pickle_name']))
            shutil.rmtree(plist['experiment_dir'])
            sys.exit()

    plist['epochs'] = 10
    plist['test_num_timesteps'] = 300
    plist['flag'] = 'train'

    return plist

def objective(trial):

    plist = my_config(trial)
    return tntt.traintest(trial, copy.deepcopy(plist))
    
study.optimize(objective, n_trials = 2)
df = study.trials_dataframe()
df.to_csv('optuna_exp.csv')
