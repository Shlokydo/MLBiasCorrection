import os
import numpy as np
import model 
import param

### config
nx     = param.param_model['dimension'] 
f      = param.param_model['forcing']
dt     = param.param_model['dt']
amp    = param.param_model['amp_const_bias']
length = param.param_exp['spinup_length']
expdir = param.param_exp['expdir']
nmem   = param.param_exp['ensembles']


### spin up of forecast model

def spinup(inum):
  np.random.seed(inum)
  x0 = np.array(np.random.randn(nx), dtype=np.float64)
  l96 = model.Lorenz96(nx,f,dt, init_x = x0, amp_const_bias = amp)
  for i in range(length):
    l96.runge_kutta()
  xtmp = np.array(l96.runge_kutta().copy(), dtype=np.float64)
  l96.save_snap(expdir + '/spinup/init'+ '{0:02d}'.format(inum) +'.nc')


os.system('mkdir -p ' +  expdir + '/spinup')
for i in range(nmem+1):  
  spinup(i)

os.system('mv ' + expdir + '/spinup/init' + '{0:02d}'.format(nmem) + '.nc ' + expdir + '/spinup/init.nc')
