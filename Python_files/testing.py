import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import argparse

import helperfunctions as helpfunc
import network_arch as net

import plotting as p

parser = argparse.ArgumentParser(description = 'Optuna Experiment Controller')

parser.add_argument("--exp_name", "-en", default = "simple_test", type = str, help = "Experiment name to test on.")
parser.add_argument("--timesteps", "-ts", default = 40, type = int, help = "Number of timesteps for testing")
args = parser.parse_args()

def test(plist, model, a_f, s_f, time_splits):

    root_grp = Dataset(plist['netCDf_loc'], "a", format="NETCDF4")

    #Extrating the datasets
    analysis_init = root_grp["vam"][25000:25000+args.timesteps]
    forecast_init = root_grp["vfm"][25000:25000+args.timesteps]

    forecast_dataset = helpfunc.locality_creator(forecast_init, plist['locality'], plist['xlocal'])
    if plist['degree'] > 1:
        if plist['locality'] == 1:
            forecast_dataset = helpfunc.make_poly(np.squeeze(forecast_dataset), plist['degree'])
            print('Poly forecast shape: ', forecast_dataset.shape)
        else:
            print('Cannot implement polynomial version as locality is not 1.')
            sys.exit()
    
    n_forecast = np.divide(np.subtract(forecast_dataset, a_f), s_f)
    c_forecast = np.zeros((analysis_init.shape[0], analysis_init.shape[1]), dtype='float32')

    if plist['make_recurrent']:
        for i in range(forecast_dataset.shape[1]):
            b_forecast = forecast_init[i + time_splits - 1,:]
            forecast = n_forecast[:, i:i + time_splits, :]
            #c_forecast[:, i + time_splits - 1] = np.add(np.multiply(np.squeeze(model(forecast).numpy()), s_a), a_a) + b_forecast[:, -1, :] 
            c_forecast[:, i + time_splits - 1] = np.squeeze(model(forecast, [])[0].numpy()) + b_forecast 
    else:
        for j in range(forecast_dataset.shape[1]):             
            b_forecast = forecast_init[j,:]
            forecast = n_forecast[:,j,:]
            c_forecast[j,:] = np.squeeze(model(forecast, [])[0].numpy()) + b_forecast

    c_rmse = np.sqrt(np.mean(np.power(analysis_init - c_forecast, 2)))
    rmse = np.sqrt(np.mean(np.power(analysis_init - forecast_init, 2)))
    print('Non-Corrected RMSE: ', rmse)
    print('Corrected RMSE: ', c_rmse)

    return c_forecast, analysis_init, forecast_init, c_rmse, rmse

def testing(plist):

    print('\nGPU Available: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    c_max = tf.Variable(1.4, dtype = tf.float32)
    c_min = tf.Variable(-3.4, dtype = tf.float32)
    a_f = tf.Variable(tf.zeros(16, dtype = tf.float32))
    s_f = tf.Variable(tf.zeros(16, dtype = tf.float32))
    time_splits = tf.Variable(0)
    model = net.rnn_model(plist, c_max, c_min)

    #Defining the checkpoint instance
    #checkpoint = tf.train.Checkpoint(model = model, a_a = a_a, s_a = s_a, a_f = a_f, s_f = s_f, time_splits = time_splits)
    checkpoint = tf.train.Checkpoint(model = model, a_f = a_f, s_f = s_f, time_splits = time_splits, c_max = c_max, c_min = c_min)

    #Creating checkpoint instance
    save_directory = plist['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, max_to_keep= plist['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    a_f = np.reshape(a_f.numpy(), (a_f.shape[0], 1, 1))
    s_f = np.reshape(s_f.numpy(), (s_f.shape[0], 1, 1))

    #print('Avg. of analysis - forecast normalized: ', a_a.numpy())
    #print('Std. of analysis - forecast normalized: ', s_a.numpy())
    #print('Avg. of forecast normalized: ', c_max)
    #print('Std. of forecast normalized: ', c_min)
    #print('Time stepping: ', time_splits.numpy())

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        print('Starting testing...')
        c_forecast, analysis, forecast, c_rmse, rmse = test(plist, model, a_f, s_f, time_splits)
        p.plot_func(plist, c_forecast, analysis, forecast, c_rmse, rmse)
        return 

    else:
        print("No checkpoint exists.")
        print('Cannot test as no checkpoint exists. Exiting...')
        return 

experiment_name = args.exp_name
pickle_fileloc = './n_experiments/' + experiment_name + '/checkpoint' + '/params.pickle'
plist = helpfunc.read_pickle(pickle_fileloc)

testing(plist)
