#------------------------------------------------
import os
import numpy as np
import netCDF4
import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 

expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
#dadir  = param.param_exp['da_type']

dadir=('nocorr','offline_linear')

#------------------------------------------------

# load nature and observation data
nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[0] + '/stats.nc','r',format='NETCDF4')
rmse = np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32)
sprd = np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

nc = netCDF4.Dataset(expdir + '/' + obsdir +'/' + dadir[1] + '/stats.nc','r',format='NETCDF4')
rmse_2 = np.array(nc.variables['rmse'][:], dtype=type(np.float64)).astype(np.float32)
sprd_2 = np.array(nc.variables['sprd'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

ntime=len(time)


rmse_plot=np.mean(rmse,axis=0)
sprd_plot=np.mean(sprd,axis=0)

rmse_2_plot=np.mean(rmse_2,axis=0)
sprd_2_plot=np.mean(sprd_2,axis=0)


doubling_time=0.2*2.1 ### 2.1 day

refy=np.array((1,max(rmse_plot)))
refx=np.array((0,doubling_time*np.log2(refy[1])))

smp_e=1
nplt=200
###
plt.figure()
plt.yscale('log')
#plt.scatter(test_labels, test_predictions)
#plt.scatter(fcst[ntime-100:ntime,1], anal[ntime-100:ntime,1]-fcst[ntime-100:ntime,1])
plt.plot(time, rmse_plot)
plt.plot(time, rmse_2_plot)
#plt.plot(time, sprd_plot)
plt.plot(refx, refy,color='black',linestyle='dashed')
plt.xlabel('time')
plt.ylabel('RMSE')
#plt.axis('equal')
#plt.axis('square')
plt.xlim()
plt.ylim()
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
plt.savefig('rmse.png')
###

