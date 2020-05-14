import os
import numpy as np
import model 
import param

### config
nx        = param.param_model['dimension'] 
f         = param.param_model['forcing']
dt        = param.param_model['dt']
length    = param.param_exp['spinup_length']
expdir    = param.param_exp['expdir']
amp       = param.param_add['amp_add_bias']
amp_2     = param.param_add['amp_add_bias_2']
bias_mode = param.param_add['bias_mode']

### spin up of coupled Lorenz96 system (nature run)

def spinup(inum):
  np.random.seed(inum)
  x0  = np.array(np.random.randn(nx),  dtype=np.float64)
  l96 = model.Lorenz96_add(nx, f ,dt, x0, amp, amp_2, bias_mode)
  for i in range(length):
    l96.runge_kutta()
  l96.save_snap(expdir + '/spinup/init_coupled.nc')

os.system('mkdir -p ' +  expdir + '/spinup')
spinup(0)
