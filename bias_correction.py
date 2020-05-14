import math                               
import numpy as np                    
import numpy.linalg as LA             
import netCDF4
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
      self.dim_y_loc = 1 + 2 * param.param_letkf['localization_length_cutoff']
      self.constb = np.zeros(dim_y, dtype=np.float64)
      self.coeffw = np.zeros((dim_y,self.dim_y_loc), dtype=np.float64)
      if param.param_bc['offline'] is not None:
        nc = netCDF4.Dataset(param.param_bc['path'],'r',format='NETCDF4')
        arrayw = np.array(nc.variables['w'][:], dtype=type(np.float64)).astype(np.float32)
        for j in range(dim_y):
          self.constb[j] = arrayw[0]
          self.coeffw[j,:] = arrayw[1:self.dim_y_loc+1,0] 
#          print(self.constb[j])
#          print(self.coeffw[j,:])
#          quit()
    if mode is 'linear_custom':
      self.dim_y_loc = 1 + 2 * param.param_letkf['localization_length_cutoff']
      self.dim_p = 1 + 3 * self.dim_y_loc 
#      self.dim_p = 1 + 2 * self.dim_y_loc 
      self.pval = np.zeros(self.dim_p,dtype=np.float64)
      self.coeffw = np.zeros((dim_y,self.dim_p), dtype=np.float64)
    if mode is 'tf':
      self.tfm = bctf.BCTF(param.param_model['dimension'])

  def custom_basisf(self,y_in_loc):
    self.pval[0]=1 ### steady component
    self.pval[1:self.dim_y_loc+1] = y_in_loc
    self.pval[1+self.dim_y_loc:2*self.dim_y_loc+1] = y_in_loc**2
    self.pval[1+2*self.dim_y_loc:3*self.dim_y_loc+1] = y_in_loc**3
#      p_out=np.append(p_out,y_in_loc[i]**2)
#      p_out=np.append(p_out,y_in_loc[i]**3)
#    return p_out

  def localize(self,y_in,indx):
    y_out=np.zeros(self.dim_y_loc)
    for i in range(self.dim_y_loc):
      iloc = indx + i - param.param_letkf['localization_length_cutoff']
      if (iloc < 0) :
        iloc += self.dim_y
      elif (iloc > self.dim_y-1):
        iloc -= self.dim_y
      y_out[i] = y_in[iloc]
    return y_out

  def train(self,y_in,y_out):
    if self.mode is None:
      return    
    elif self.mode is 'const':
      self.constb = (1-self.gamma) * self.constb 
      self.constb = self.constb + self.alpha * (y_out - y_in)
    elif self.mode is 'linear':
      self.constb = (1-self.gamma) * self.constb 
      self.coeffw = (1-self.gamma) * self.coeffw 
      for j in range(self.dim_y):
        self.constb[j] = self.constb[j] + self.alpha * (y_out[j] - y_in[j])
        self.coeffw[j] = self.coeffw[j] + self.alpha * self.localize(y_in,j) * (y_out[j] - y_in[j]) / (1 + (self.localize(y_in,j)**2).sum()) 
    elif self.mode is 'linear_custom':
      self.coeffw = (1-self.gamma) * self.coeffw 
      for j in range(self.dim_y):
        self.custom_basisf(self.localize(y_in,j))   
        self.coeffw[j] = self.coeffw[j] + self.alpha * self.pval * (y_out[j] - y_in[j]) / (1 + (self.pval**2).sum()) 
  
 
  def correct(self,y_in):  
    if self.mode is None:
      y_out = y_in 
    elif self.mode is 'const':
      y_out = y_in + self.constb
    elif self.mode is 'linear':
      y_out = np.zeros(self.dim_y,dtype=np.float64)
      for j in range(self.dim_y):
        y_out[j] = y_in[j] + self.constb[j] + np.dot(self.coeffw[j],self.localize(y_in,j))
    elif self.mode is 'linear_custom':
      y_out = np.zeros(self.dim_y,dtype=np.float64)
      for j in range(self.dim_y):
        self.custom_basisf(self.localize(y_in,j))   
        y_out[j] = y_in[j] + np.dot(self.coeffw[j],self.pval)
    elif self.mode is 'tf':
###      y_in = self.tfm.locality_gen(y_in)
      y_out = self.tfm.predict(y_in)
    return y_out
