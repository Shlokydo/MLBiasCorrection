#------------------------------------------------
import os
import sys
import math
import numpy as np
import numpy.linalg as LA
import netCDF4
import model
import letkf
import param
import bias_correction as BC

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

nmem        = param.param_exp['ensembles']
dt_nature   = param.param_exp['dt_nature']
dt_assim    = param.param_exp['dt_assim']

loc_scale  = param.param_letkf['localization_length_scale']
loc_cutoff = param.param_letkf['localization_length_cutoff']
fact_infl  = param.param_letkf['inflation_factor'] 


bc_type = param.param_bc['bc_type']
bc_alpha = param.param_bc['alpha']
bc_gamma = param.param_bc['gamma']

intv_nature=int(dt_nature/dt)

inittime=int(sys.argv[1])
length=200

#------------------------------------------------

letkf = letkf.LETKF(model.Lorenz96, nx, f, amp_const_bias = amp, k = nmem, localization_len = loc_scale, localization_cut = loc_cutoff , inflation = fact_infl)

nc = netCDF4.Dataset(expdir + '/nature.nc','r',format='NETCDF4')
nature = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time_nature = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

# initial ensemble perturbation
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/assim.nc','r',format='NETCDF4')
analysis = np.array(nc.variables['va'][:], dtype=type(np.float64)).astype(np.float32)
time_assim = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close
for i in range(nmem):
  letkf.ensemble[i].x = analysis[inittime,i,:]

inittime_nature = int(np.where(time_nature == time_assim[inittime])[0])
ntime_nature=len(time_nature)

bc=BC.BiasCorrection(bc_type,nx,bc_alpha,bc_gamma)

xf=[]
xfm=[]
xfm_raw=[]
rmse=[]
sprd=[]
# MAIN LOOP
#for step in range(inittime_nature, ntime_nature):
for step in range(inittime_nature, inittime_nature+length):
  for i in range(intv_nature):
    letkf.forward()
  
  xfmtemp=letkf.mean()
  xfm_raw.append(xfmtemp)
  for i in range(nmem):
    letkf.ensemble[i].x = bc.correct(letkf.ensemble[i].x)
  xf.append(letkf.members())
  xfmtemp=letkf.mean()
  xfm.append(xfmtemp)

  rmse.append(math.sqrt(((letkf.mean()-nature[step+1])**2).sum() /nx))
  sprd.append(math.sqrt((letkf.sprd()**2).sum()/nx))
#  if ( round(icount/10,4).is_integer() ):
#  print("time {0:10.4f}, RMSE , {1:8.4f}, SPRD ,{2:8.4f}".format(round(time_nature[step],4),rmse[step-inittime_nature],sprd[step-inittime_nature]))
        
#print("done")


xf  = np.array(xf,  dtype=np.float64)
xfm_raw = np.array(xfm_raw, dtype=np.float64)
xfm = np.array(xfm, dtype=np.float64)
rmse  = np.array(rmse,  dtype=np.float64)
sprd  = np.array(sprd,  dtype=np.float64)

os.system('mkdir -p ' + expdir + '/' + obsdir + '/' + dadir)
 
nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/forecast.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('x',nx)
nc.createDimension('e',nmem)
nc.createDimension('t',length)
x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
e_in = nc.createVariable('e',np.dtype('float64').char,('e'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
v_in = nc.createVariable('v',np.dtype('float64').char,('t','x'))
vf_in = nc.createVariable('vf',np.dtype('float64').char,('t','e','x'))
vfm_in = nc.createVariable('vfm',np.dtype('float64').char,('t','x'))
vfm_raw_in = nc.createVariable('vfm_raw',np.dtype('float64').char,('t','x'))
x_in[:] = np.array(range(1,1+nx))
e_in[:] = np.array(range(1,1+nmem))
t_in[:] = np.round(time_nature[:length],4)
v_in[:,:] = nature[:length]
vf_in[:,:,:] = xf
vfm_in[:,:] = xfm
vfm_raw_in[:,:] = xfm_raw
nc.close 

if (os.path.isfile(expdir + '/' + obsdir + '/' + dadir + '/stats.nc')) :
  nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats.nc','r+',format='NETCDF3_CLASSIC')
  nsmp=len(nc.variables['rmse'])
  nc.variables['rmse'][nsmp,:]=rmse
  nc.variables['sprd'][nsmp,:]=sprd
  nc.close
else : 
  nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats.nc','w',format='NETCDF3_CLASSIC')
  nc.createDimension('t',length)
  nc.createDimension('s',None)
  rmse_in = nc.createVariable('rmse',np.dtype('float64').char,('s','t'))
  sprd_in = nc.createVariable('sprd',np.dtype('float64').char,('s','t'))
  t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
  rmse_in[0,:] = rmse
  sprd_in[0,:] = sprd
  t_in[:] = np.round(time_nature[:length],4)
  nc.close



