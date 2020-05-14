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
  pickle_name = './n_experiments/obs001_dt005_L2D_tanh_3_best/checkpoint/params.pickle' #Enter the location of the parameter_list pickle file name
  parameter_list = helpfunc.read_pickle(pickle_name)
  return parameter_list

def get_model(plist):
  
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

class BCTF():
  
  def __init__(self, num_var):
    self.plist = get_pickle() #Getting the parameter_list
    self.model, self.time_splits = get_model(self.plist)  #Getting the model
    self.dim = num_var
    self.inp_counter = 1
    self.network_array = np.zeros((self.dim, self.time_splits, self.plist['locality']))

  def locality_gen(self, inp):
    inp = np.expand_dims(inp, axis = 0)
    a = helpfunc.locality_creator(inp, self.plist['locality'], self.plist['xlocal'])
    return a

  def predict(self, inp):
    
    @tf.function
    def pred(fore):
      return self.model(fore)
    
    inp_locality = self.locality_gen(inp)
    
    if self.inp_counter == self.time_splits:
      self.network_array = np.roll(self.network_array, -1, axis = 0) 
      self.network_array[:,-1,:] = np.squeeze(inp_locality)
      return np.squeeze(self.model(self.network_array, [])[0].numpy()) + inp 
    else:
      self.network_array[:,self.inp_counter,:] = np.squeeze(inp_locality)
      self.inp_counter += 1
      return inp
