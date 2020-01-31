import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import pickle

import new_network_arch as net
import helperfunctions as helpfunc


def get_pickle():
  pickle_name = './n_experiments/L15_D10_5/checkpoint/params.pickle' #Enter the location of the parameter_list pickle file name
  parameter_list = helpfunc.read_pickle(pickle_name)
  return parameter_list

def get_model(*args):
  
  parameter_list = args[0]
  print('\nGetting the Tensorflow model\n')
  parameter_list['stateful'] = True
  model = net.rnn_model(parameter_list)

  #Creating checkpoint instance
  checkpoint = tf.train.Checkpoint(epoch = tf.Variable(0), model = model)
  manager = tf.train.CheckpointManager(checkpoint, directory= parameter_list['checkpoint_dir'], 
                                      max_to_keep= parameter_list['max_checkpoint_keep'])
  checkpoint.restore(manager.latest_checkpoint).expect_partial()

  #Checking if previous checkpoint exists
  if manager.latest_checkpoint:
    print("Restored model from {}".format(manager.latest_checkpoint))

  return model

def prediction(*args):
 
  parameter_list = args[0]
  model = args[1]
  input_array = np.zeros((1,args[2].shape[0]))
  input_array[0,:] = args[2]
  new_forecast = np.zeros((input_array.shape[-1]))
  forecast_data = helpfunc.locality_creator(input_array, parameter_list['locality'], parameter_list['xlocal'])
  forecast_data = np.transpose(forecast_data, axes=(1,0,2))

  for i in range(forecast_data.shape[1]):
    forecast = np.expand_dims(forecast_data[:,i,:], axis = 1)
    new_forecast[i] = np.squeeze(model(forecast).numpy())
  
  return new_forecast
