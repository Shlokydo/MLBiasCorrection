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

class BCTF():
  
  def __init__(self, num_var):
    self.plist = self.get_pickle() #Getting the parameter_list
    self.model, self.time_splits = self.get_model(self.plist)  #Getting the model
    self.dim = num_var * 20 
    self.num_var = num_var
    self.inp_counter = 1
    self.network_array = np.zeros((self.dim, self.time_splits, self.plist['locality']))
    self.inp_locality = np.zeros((self.dim, 1, self.plist['locality']))

  def get_pickle(self):
    pickle_name = './n_experiments/obs001_dt020_LD_tanh5_5_best/checkpoint/params.pickle' #Enter the location of the parameter_list pickle file name
#    pickle_name = './n_experiments/slow_sint_0_CLD_tanh10_5_best/checkpoint/params.pickle' #Enter the location of the parameter_list pickle file name
    parameter_list = helpfunc.read_pickle(pickle_name)

#   print(parameter_list)
#    quit()
    return parameter_list

  def get_model(self, plist):
    
    parameter_list = plist
    print('\nGetting the Tensorflow model\n')
    #Get the Model
    a_f = tf.Variable(tf.zeros(16, dtype = tf.float32))
    s_f = tf.Variable(tf.zeros(16, dtype = tf.float32))
    time_splits = tf.Variable(0)
    model = net.rnn_model(parameter_list)

    #Creating checkpoint instance
    checkpoint = tf.train.Checkpoint(model = model, a_f = a_f, s_f = s_f, time_splits = time_splits)
    manager = tf.train.CheckpointManager(checkpoint, directory= parameter_list['checkpoint_dir'], 
                                        max_to_keep= parameter_list['max_checkpoint_keep'])
    checkpoint.restore(manager.latest_checkpoint).expect_partial()

    #Checking if previous checkpoint exists
    if manager.latest_checkpoint:
      print("Restored model from {}".format(manager.latest_checkpoint))

    return model, time_splits.numpy()

  def locality_gen(self, inp):
    inp = np.expand_dims(inp, axis = 0)
    a = helpfunc.locality_creator(inp, self.plist['locality'], self.plist['xlocal'])
    return a
  
  @tf.function
  def pred(self, fore):
    return self.model(fore, [])

  def predict(self, inp):
    
    for i in range(20):
      self.inp_locality[i*self.num_var:(i+1)*self.num_var] = self.locality_gen(inp[i*self.num_var:(i+1)*self.num_var])
    
    if self.inp_counter == self.time_splits:
      self.network_array = np.roll(self.network_array, -1, axis = 1)  #Here was the faulty code as axis argument was initally 0, creating faulty inputs for LSTM case.
      self.network_array[:,-1,:] = np.squeeze(self.inp_locality, 1)        #Dense case was doing OK because all the Variable dimension was updated in this step. 
      return np.squeeze(self.pred(self.network_array)[0].numpy()) + inp 
    else:
      self.network_array[:,self.inp_counter,:] = np.squeeze(self.inp_locality, 1)
      self.inp_counter += 1
      return inp
