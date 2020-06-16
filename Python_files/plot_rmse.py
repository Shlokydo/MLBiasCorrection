#------------------------------------------------
import os
import numpy as np
import netCDF4
import param
import matplotlib.pyplot as plt

from plotting import plot_func_amemiya_ef

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 

expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
#dadir  = param.param_exp['da_type']

dadir=('nocorr','linear_offline','test_tf')

#------------------------------------------------

nexp=len(dadir)

rmse=[]
sprd=[]
rmse_plot=[]
sprd_plot=[]


# load nature and observation data
for i in range(nexp):
  nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[i] + '/stats.nc','r',format='NETCDF4')
  rmse.append(np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32))
  sprd.append(np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32))
  if (i == 0): 
    time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
  nc.close 
  rmse_plot.append(np.mean(rmse[i],axis=0))
  sprd_plot.append(np.mean(sprd[i],axis=0))

  plot_func_amemiya_ef(rmse_plot, sprd_plot, time, dadir, exp_name = 'Ext_Forecast', run_name = 'test') #Please change the run_name argument to the Case you are currently plotting for, i.e. 'advection', 'sparseTime', 'stepBias' and likewise
