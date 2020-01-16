import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys

import helperfunctions as helpfunc
import new_network_arch as net

def test(parameter_list, model):

    root_grp = Dataset(parameter_list['netCDf_loc'], "a", format="NETCDF4")

    #Extrating the datasets
    analysis_init = root_grp["vam"]
    forecast_init = root_grp["vfm"]

    analysis_dataset = helpfunc.truth_label_creator(analysis_init[10:parameter_list['test_num_timesteps']])
    forecast_dataset = helpfunc.locality_creator(forecast_init[10:parameter_list['test_num_timesteps']],
                                                            parameter_list['locality'],
                                                            parameter_list['xlocal'])
    
    new_forecast = np.zeros((analysis_dataset.shape[0], analysis_dataset.shape[1]), dtype='float32')

    if parameter_list['make_recurrent']:
        for i in range(forecast_dataset.shape[1]):
            forecast = np.expand_dims(forecast_dataset[:,i,:], axis=1)
            new_forecast[:,i] = np.squeeze(model(forecast))
        new_forecast = np.transpose(new_forecast)
    else:
        for j in range(forecast_dataset.shape[0]):             
                forecast = forecast_dataset[j,:,:]
                new_forecast[j,:,:] = model(forecast)
        new_forecast = np.transpose(np.squeeze(new_forecast))

    test_time = root_grp.createDimension('tt', forecast_dataset.shape[1])
    v_test_time = root_grp.createVariable('v_test_time', 'i', ('tt',))
    model_vfm = root_grp.createVariable(parameter_list['experiment_name'] + '_vfm', 'f4', ('tt','x',))
    model_vfm[:] = new_forecast
    root_grp.close()

def testing(parameter_list):

    parameter_list['stateful'] = True
    print('\nGPU Available: {}\n'.format(tf.test.is_gpu_available()))

    #Get the Model
    # with mirrored_strategy.scope():
    model = net.rnn_model(parameter_list)

    #Defining the checkpoint instance
    checkpoint = tf.train.Checkpoint(model = model)

    #Creating summary writer
    summary_writer = tf.summary.create_file_writer(logdir= parameter_list['log_dir'])

    #Creating checkpoint instance
    save_directory = parameter_list['checkpoint_dir']
    manager = tf.train.CheckpointManager(checkpoint, directory= save_directory, 
                                        max_to_keep= parameter_list['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

        print('Starting testing...')
        test(parameter_list, model)
        return parameter_list['global_epoch']

    else:
        print("No checkpoint exists.")
        print('Cannot test as no checkpoint exists. Exiting...')
        return parameter_list['global_epoch']
