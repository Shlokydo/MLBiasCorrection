import copy
import random
import numpy as np
import pandas as pd
import os
import sys

import helperfunctions as helpfunc
import training_test as tntt
import testing as tes
import tensorflow as tf

from sacred import Experiment
from sacred.observers import MongoObserver

exp = Experiment('Bias_Correction')
#exp.observers.append(MongoObserver(url = 'mongodb://localhost:27017', db_name = 'mlbias'))

@exp.config
def my_config():
    #Parameter List
    plist = {}

    #Dataset and directories related settings
    plist['netCDf_loc'] = "/home/amemiya/assim.nc"
    plist['xlocal'] = 3
    plist['locality'] = 19
    plist['num_timesteps'] = 30
    plist['time_splits'] = 30
    plist['tfrecord_analysis'] = './tfrecord/X40F18_I20_analysis.tfrecord'
    plist['tfrecord_forecast'] = './tfrecord/X40F18_I20_forecast.tfrecord'
    plist['experiment_name'] = "L15_D10_5"
    plist['experiment_dir'] = './n_experiments/' + plist['experiment_name'] 
    plist['checkpoint_dir'] = plist['experiment_dir'] + '/checkpoint'
    plist['log_dir'] = plist['experiment_dir'] + '/log'

    plist['pickle_name'] = plist['checkpoint_dir'] + '/params.pickle'

    if not os.path.exists(plist['experiment_dir']):
        os.makedirs(plist['log_dir'])
        os.mkdir(plist['checkpoint_dir'])

        #Network related settings
        plist['make_recurrent'] = True 
        plist['num_lstm_layers'] = 1
        plist['num_dense_layers'] = 3
        plist['dense_output'] = [10, 5]
        plist['LSTM_output'] = [15]
        plist['net_output'] = 1
        plist['activation'] = 'tanh'
        plist['rec_activation'] = 'hard_sigmoid'
        plist['l2_regu'] = 0.0
        plist['l1_regu'] = 0.0
        plist['lstm_dropout'] = 0.0
        plist['rec_lstm_dropout'] = 0.0
        plist['unroll_lstm'] = False
        plist['new_forecast'] = False 
        plist['stateful'] = False

        #Training related settings
        plist['max_checkpoint_keep'] = 1
        plist['log_freq'] = 5
        plist['early_stop_patience'] = 500
        plist['num_epochs_checkpoint'] = 1
        plist['summery_freq'] = 1
        plist['global_epoch'] = 0
        plist['global_batch_size'] = 1024 
        plist['val_size'] = 1
        plist['lr_decay_steps'] = 300000
        plist['lr_decay_rate'] = 0.70
        try:
            plist['learning_rate'] = 1e-3 * plist['global_batch_size'] / (256 * len(tf.config.experimental.list_physical_devices('GPU')))
        except:
            plist['learning_rate'] = 1e-3 * plist['global_batch_size'] / (256)
        plist['val_min'] = 1000

    else:
        if os.path.isfile(plist['pickle_name']):
            plist = helpfunc.read_pickle(plist['pickle_name'])
        else:
            print('\nNo pickle file exists at {}. Exiting....\n'.format(plist['pickle_name']))
            sys.exit()

    plist['epochs'] = 10
    plist['test_num_timesteps'] = 300
    plist['flag'] = 'train'

@exp.automain
def mainfunc(plist):
    if plist['flag'] == 'train':
        plist['global_epoch'], plist['val_min'] = tntt.traintest(copy.deepcopy(plist))
    else:
        plist['global_epoch'] = tes.testing(copy.deepcopy(plist))
