import math                               
import numpy as np                    
import numpy.linalg as LA             
import bias_correction_tf as bctf
import param

class BiasCorrection:
  def __init__(self, mode=None, dim_y = 0 , alpha = 0, gamma = 0):
    self.mode=mode
    if mode is not None:
      self.alpha = alpha
      self.gamma = gamma
      self.dim_y = dim_y
    if mode is 'const':
      self.constb = np.zeros(dim_y, dtype=np.float64)
    if mode is 'linear':
      self.constb = np.zeros(dim_y, dtype=np.float64)
      self.coeffw = np.zeros((dim_y,dim_y), dtype=np.float64)
    if mode is 'tf':
      self.tfm = bctf.BCTF(param.param_model['dimension'])
  
  def train(self,y_in,y_out):
    if self.mode is None:
      return    
    elif self.mode is 'const':
      self.constb = (1-self.gamma) * self.constb 
      self.constb = self.constb + self.alpha * (y_out - y_in)
    elif self.mode is 'linear':
      self.constb = (1-self.gamma) * self.constb 
      self.coeffw = (1-self.gamma) * self.coeffw 
      self.constb = self.constb + self.alpha * (y_out - y_in)
      for j in range(self.dim_y):
        self.coeffw[j] = self.coeffw[j] + self.alpha * y_in[j] * (y_out - y_in) / (1 + (y_in**2).sum()) 
#        self.coeffw[j] = self.coeffw[j] + self.alpha * y_in[j] * (y_out - y_in) 
 
 
  def correct(self,y_in):  
    if self.mode is None:
      y_out = y_in 
    elif self.mode is 'const':
      y_out = y_in + self.constb
    elif self.mode is 'linear':
      y_out = np.zeros(self.dim_y,dtype=np.float64)
      for j in range(self.dim_y):
        y_out[j] = y_in[j] + self.constb[j] + np.dot(self.coeffw[j],y_in)
    elif self.mode is 'tf':
      y_in = self.tfm.locality_gen(y_in)
      y_out = self.tfm.predict(y_in)
    return y_out
