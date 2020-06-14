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

#inittime=int(sys.argv[1])
length=200
nsmpmax=20
intv=250
inittime=4500+250   ### spinup

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

assim_length=len(time_assim)

ismp=0
rmsesmp=[]
sprdsmp=[]
while ((inittime+length <= assim_length) and (ismp < nsmpmax)) :
  
  print("inittime ",inittime)
  
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

# init value
  xf.append(letkf.members())
  xfm.append(letkf.mean())
  xfm_raw.append(letkf.mean())
  rmse.append(math.sqrt(((letkf.mean()-nature[inittime_nature])**2).sum() /nx))
  sprd.append(math.sqrt((letkf.sprd()**2).sum()/nx))
# 

# MAIN LOOP
  for step in range(inittime_nature, inittime_nature+length):
    for i in range(intv_nature):
      letkf.forward()
      if param.param_bc['correct_step'] is not None:
        for i in range(nmem):
          fact=1.0/intv_assim
          letkf.ensemble[i].x = fact * bc.correct(letkf.ensemble[i].x) + (1.0-fact) * letkf.ensemble[i].x
   
    xfmtemp=letkf.mean()
    xfm_raw.append(xfmtemp)
    if param.param_bc['correct_step'] is not None:
       for i in range(nmem):
            fact=1.0/intv_assim
            letkf.ensemble[i].x = fact * bc.correct(letkf.ensemble[i].x) + (1.0-fact) * letkf.ensemble[i].x
    else:
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

  rmsesmp.append(rmse)
  sprdsmp.append(sprd)

  ismp+=1 
  inittime+=intv

rmsesmp  = np.array(rmsesmp,  dtype=np.float64)
sprdsmp  = np.array(sprdsmp,  dtype=np.float64)

if (os.path.isfile(expdir + '/' + obsdir + '/' + dadir + '/stats.nc')) :
    print('overwrite')
    nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats.nc','r+',format='NETCDF3_CLASSIC')
    nsmp=len(nc.variables['rmse'])
    nc.variables['rmse'][nsmp:,:]=rmsesmp
    nc.variables['sprd'][nsmp:,:]=sprdsmp
    nc.close
else : 
    print('create')
    nc = netCDF4.Dataset(expdir + '/' + obsdir + '/' + dadir + '/stats.nc','w',format='NETCDF3_CLASSIC')
    nc.createDimension('t',length+1)
    nc.createDimension('s',None)
    rmse_in = nc.createVariable('rmse',np.dtype('float64').char,('s','t'))
    sprd_in = nc.createVariable('sprd',np.dtype('float64').char,('s','t'))
    t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
    rmse_in[:,:] = rmsesmp
    sprd_in[:,:] = sprdsmp
    t_in[:] = np.round(time_nature[:length+1],4)
    nc.close


