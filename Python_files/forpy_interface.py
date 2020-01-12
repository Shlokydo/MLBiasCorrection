import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import pickle

import helperfuntions as helpfunc
import new_network_arch as net

pickle_name = '' #Enter the location of the parameter_list pickle file name
parameter_list = helpfunc.read_pickle(pickle_name)

def get_model():
  
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
  
  model = args[0]
  new_forecast = np.zeros(args[1].shape[1])
  forecast_data = helpfunc.locality_creator(args[1], parameter_list['locality'], parameter_list['x_local'])

  for i in range(forecast_data.shape[1]):
    forecast = np.expand_dims(forecast_data[:,i,:], axis = 1)
    new_forecast[i] = np.squeeze(model(forecast))
  
  return np.squeeze(new_forecast)