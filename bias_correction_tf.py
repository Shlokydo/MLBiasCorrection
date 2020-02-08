import tensorflow as tf
import numpy as np
from netCDF4 import Dataset
import time
import math
import os
import sys
import pickle

import network_arch as net
import helperfunctions as helpfunc

def get_pickle():
  pickle_name = './n_experiments/L15_D10_5/checkpoint/params.pickle' #Enter the location of the parameter_list pickle file name
  parameter_list = helpfunc.read_pickle(pickle_name)
  return parameter_list

def get_model(plist):
  
  parameter_list = plist
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

class BCTF():
  
  def __init__(self, num_var):
    self.plist = get_pickle() #Getting the parameter_list
    self.model = get_model(self.plist)  #Getting the model
    sta_indi = [tf.zeros((1, self.plist['LSTM_output'][i]), tf.float32) for i in range(self.plist['num_lstm_layers'])]
    self.state_h = [sta_indi for i in range(num_var)]
    self.state_c = [sta_indi for i in range(num_var)]
    self.dim = num_var

  def locality_gen(self, inp):
    inp = np.expand_dims(inp, axis = 0)
    a = helpfunc.locality_creator(inp, self.plist['locality'], self.plist['xlocal'])
    return np.transpose(a, axes=(1,0,2))

  def predict(self, inp):
    
    @tf.function
    def pred(fore, st):
      return self.model(fore, st)

    new_forecast = tf.TensorArray(tf.float32, size = inp.shape[1], element_shape=(1, 1, 1))

    for i in range(inp.shape[1]):
      forecast = tf.expand_dims(inp[:,i,:], axis = 1)
      new, self.state_h[i], self.state_c[i] = pred(forecast, [self.state_h[i], self.state_c[i]])
      new_forecast.write(i, new)
   
    new_forecast = new_forecast.stack()
    new_forecast = tf.squeeze(new_forecast)
    return new_forecast.numpy()
