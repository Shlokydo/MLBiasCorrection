#------------------------------------------------
import os
import math
import numpy as np
import numpy.linalg as LA
import netCDF4
import model
import letkf
import param

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 
f   = param.param_model['forcing']
dt  = param.param_model['dt']
amp = param.param_model['amp_const_bias']

length = param.param_exp['exp_length']
expdir = param.param_exp['expdir']
obsdir = param.param_exp['obs_type']
dadir  = param.param_exp['da_type']

obs_err_std = param.param_letkf['obs_error']
nmem        = param.param_exp['ensembles']
dt_nature   = param.param_exp['dt_nature']
dt_obs      = param.param_exp['dt_obs']
dt_assim    = param.param_exp['dt_assim']

loc_scale  = param.param_letkf['localization_length_scale']
loc_cutoff = param.param_letkf['localization_length_cutoff']
fact_infl  = param.param_letkf['inflation_factor'] 

intv_nature=int(dt_nature/dt)
#------------------------------------------------


letkf = letkf.LETKF(model.Lorenz96, nx, k = nmem, localization_len = loc_scale, localization_cut = loc_cutoff , inflation = fact_infl)
# initial ensemble perturbation
for i in range(nmem):
  nc = netCDF4.Dataset(expdir + '/spinup/init'+ '{0:02d}'.format(i) + '.nc','r',format='NETCDF4')
  x0 = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
  nc.close 
  letkf.ensemble[i].x = x0

# load nature and observation data
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/obs.nc','r',format='NETCDF4')
obs = np.array(nc.variables['vy'][:], dtype=type(np.float64)).astype(np.float32)
time_obs = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

nc = netCDF4.Dataset(expdir + '/nature.nc','r',format='NETCDF4')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time_nature = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

# set observation error covariance matrix (diagonal elements only)
r = np.ones(nx, dtype=np.float64) * obs_err_std
# set observation operator matrix (use identity)
h = np.identity(nx, dtype=np.float64)

xfm = []
xam = []
xf  = []
xa  = []
time_assim = []
ntime_nature=len(time_nature)

# MAIN LOOP
for step in range(ntime_nature):
  for i in range(intv_nature):
    letkf.forward()
  if (np.count_nonzero(time_obs == time_nature[step])): 
    step_obs=int(np.where(time_obs == time_nature[step])[0])
    if (round(time_obs[step_obs]/dt_assim,4).is_integer()):  
      if (step + 1) < length:
#        if step % 100 == 0:
#          print("analysis ", step)
#        else:
#          print(".", end="")
        xf.append(letkf.members())
        xfm.append(letkf.mean())
        xa.append(letkf.analysis(h, obs[step_obs], r))
        xam.append(letkf.mean())
        time_assim.append(round(time_obs[step_obs],4))
        print('time ', round(time_obs[step_obs],4),' RMSE ', math.sqrt(((letkf.mean()-nature[step])**2).sum() /nx))
       
        

time_assim = np.array(time_assim, dtype=np.float64)
xf  = np.array(xf,  dtype=np.float64)
xa  = np.array(xa,  dtype=np.float64)
xfm = np.array(xfm, dtype=np.float64)
xam = np.array(xam, dtype=np.float64)

print("done")

#rmse = []
#for xx in x_letkf10:
#  if xx[0] < nature.shape[0]:
#    rmse.append((xx[0], math.sqrt(((nature[xx[0]] - xx[1]) ** 2).sum() / n)))
#rmse = np.array(rmse) 

os.system('mkdir -p ' + expdir + '/' + obsdir + '/' + dadir)
 
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/assim.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('x',nx)
nc.createDimension('e',nmem)
nc.createDimension('t',None)
x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
e_in = nc.createVariable('e',np.dtype('float64').char,('e'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
va_in = nc.createVariable('va',np.dtype('float64').char,('t','e','x'))
vf_in = nc.createVariable('vf',np.dtype('float64').char,('t','e','x'))
vam_in = nc.createVariable('vam',np.dtype('float64').char,('t','x'))
vfm_in = nc.createVariable('vfm',np.dtype('float64').char,('t','x'))
x_in[:] = np.array(range(1,1+nx))
e_in[:] = np.array(range(1,1+nmem))
t_in[:] = np.round(time_assim,4)
va_in[:,:] = xa
vf_in[:,:] = xf
vam_in[:,:] = xam
vfm_in[:,:] = xfm
nc.close 


