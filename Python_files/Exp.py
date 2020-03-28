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
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import optuna

parser = argparse.ArgumentParser(description = 'Optuna Experiment Controller')

parser.add_argument("--t", default = "optimize", type = str, choices = ["optimize", "best", "p_best"], help = "Choose between optimization or training best parameter")
parser.add_argument("--m_recu", "-mr", default = 1, type = int, choices = [0, 1], help = "Use LSTM layers or not")
parser.add_argument("--epochs", "-e",  default = 200, type = int, help = "Num of epochs for training")
parser.add_argument("--num_trials", "-nt",  default = 10, type = int, help = "Num of Optuna trials")
parser.add_argument("--netcdf_dataset", "-ncdf_loc", default = "../DATA/simple_test/test_obs/test_da/assim.nc", type = str, help = "Location of the netCDF dataset")
parser.add_argument("--optuna_study", "-os", default = "simple_test", type = str, help = "Optuna Study name")
parser.add_argument("--locality", "-l",  default = 9, type = int, help = "Locality size (including the main variable)")
parser.add_argument("--degree", "-d",  default = 1, type = int, help = "To make a polynomial input")
parser.add_argument("--normalized", "-norm",  default = 1, type = int, choices = [0, 1], help = "Use normalized dataset for training.")
parser.add_argument("--time_splits", "-ts",  default = 5, type = int, help = "Num of RNN timesteps")
parser.add_argument("--train_batch", "-tb", default = 16384, type = int, help = "Training batch size")
parser.add_argument("--val_batch", "-vb", default = 16384, type = int, help = "Validation batch size")
parser.add_argument("--num_batches", "-nbs",  default = 1, type = int, help = "Number of training batch per epoch")
args = parser.parse_args()

if (args.locality > 1) and (args.degree > 1):
    parser.error('If degree is > 1 then locality has to be 1.')

def my_config(trial):
    #Parameter List
    plist = {}
    
    #Network related settings
    plist['make_recurrent'] = args.m_recu 

    if args.m_recu:
        plist['time_splits'] = args.time_splits
        #plist['time_splits'] = trial.suggest_categorical('Time_splits', [2, 3, 4, 5, 6])
        print('\nNetwork is recurrent\n')
        plist['num_lstm_layers'] = trial.suggest_int('lstm_layers', 1, 1)
        plist['LSTM_output'] = []
        plist['LSTM_output'].append(trial.suggest_int('lstm_' + str(0), 5, 25))
        for i in range(plist['num_lstm_layers'] - 1):
            plist['LSTM_output'].append(trial.suggest_int('lstm_' + str(i+1), 20, plist['LSTM_output'][i]))
    else:
        plist['time_splits'] = 1 
        print('\nNetwork is only dense\n')
        plist['num_lstm_layers'] = 0
        plist['LSTM_output'] = []

    plist['num_dense_layers'] = trial.suggest_int('dense_layers', 1, 2) 
    plist['dense_output'] = []
    plist['dense_output'].append(trial.suggest_int('dense_' + str(0), 8, 18))
    for i in range(plist['num_dense_layers'] - 1):
        plist['dense_output'].append(trial.suggest_int('dense_' + str(i+1), 4, plist['dense_output'][i]))
    plist['dense_output'].append(1)

    plist['activation'] = 'tanh'
    plist['d_activation'] = None
    plist['rec_activation'] = 'sigmoid'
    plist['l2_regu'] = 0.0
    plist['l1_regu'] = 0.0
    plist['lstm_dropout'] = 0.0
    plist['rec_lstm_dropout'] = 0.0

    #Training related settings
    plist['max_checkpoint_keep'] = 3
    plist['log_freq'] = 1
    plist['early_stop_patience'] = 400
    plist['summery_freq'] = 1
    plist['global_epoch'] = 0
    plist['global_batch_size'] = args.train_batch  
    plist['global_batch_size_v'] = args.val_batch
    plist['val_size'] = 1 * plist['global_batch_size_v']
    plist['num_timesteps'] = int(((plist['global_batch_size'] * args.num_batches + plist['val_size']) * plist['time_splits'])/ 16 + 100)
    plist['val_min'] = 1000

    plist['lr_decay_steps'] = 1000
    #plist['lr_decay_rate'] = trial.suggest_categorical('lrdr', [0.85, 0.95, 0.9])
    plist['lr_decay_rate'] = 0
    #grad_mellow = trial.suggest_categorical('grad_mellow', [1, 0.1])
    plist['grad_mellow'] = 1
    #plist['learning_rate'] = trial.suggest_loguniform('lr', 1.0e-3, 1.9e-3)
    plist['learning_rate'] = 1.0e-3

    try:
        plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128 * len(tf.config.experimental.list_physical_devices('GPU')))
    except:
        plist['learning_rate'] = plist['learning_rate'] * plist['global_batch_size'] / (128)
    
    #Dataset and directories related settings
    plist['netCDf_loc'] = args.netcdf_dataset
    plist['xlocal'] = 3
    plist['locality'] = args.locality
    plist['degree'] = args.degree
    plist['normalized'] = args.normalized
    #plist['locality'] = trial.suggest_categorical('locality', [1, 3, 5, 7])

    if args.t == 'optimize':
        plist['experiment_name'] = str(trial.study.study_name) + '_' + str(trial.number)
    elif args.t == 'best':
        plist['experiment_name'] = args.optuna_study + '_best' 

    plist['experiment_dir'] = './n_experiments/' + plist['experiment_name'] 
    plist['checkpoint_dir'] = plist['experiment_dir'] + '/checkpoint'
    plist['log_dir'] = plist['experiment_dir'] + '/log'

    plist['pickle_name'] = plist['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(plist['experiment_dir']):
        print('\nCreating experiment_dir and the corresponding sub-directories.\n')
        os.makedirs(plist['log_dir'])
        os.mkdir(plist['checkpoint_dir'])
    else:
        if os.path.isfile(plist['pickle_name']):
            print('\nPickle file exists. Reading parameter list from it.\n')
            plist = helpfunc.read_pickle(plist['pickle_name'])
        else:
            print('\nNo pickle file exists at {}. Exiting....\n'.format(plist['pickle_name']))
            #shutil.rmtree(plist['experiment_dir'])
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
    elif args.t == 'p_best':
        print('\nBest Trial number: ', study.best_trial.number)
        print('Val RMSE for the trial: ', study.best_value)
        print('Best trial parameters: ', study.best_params, '\n')
    else:
        print('\nPlease select what you want to do from.\n')
        sys.exit()

