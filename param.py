###
### Common experiment parameter
###

param_model = {}
param_model['dimension'] = 16
param_model['dimension_coupled'] = 256
param_model['forcing'] = 8
param_model['dt'] = 0.005
param_model['dt_coupled'] = 0.0001  # dt/c
param_model['h'] = 1
param_model['b'] = 20
param_model['c'] = 50
param_model['amp_const_bias'] = 0

param_letkf = {}
param_letkf['obs_error'] = 0.01
param_letkf['localization_length_scale'] = 3
param_letkf['localization_length_cutoff'] = 2
param_letkf['inflation_factor'] = 3.0

param_exp = {}
param_exp['exp_length'] = 3000                   ### replace this with larger length
param_exp['ensembles'] = 20
param_exp['expdir'] = './DATA/test_step2'
param_exp['obs_type'] = 'obs_001'
param_exp['da_type'] = 'offline_linear_IAU'
param_exp['dt_nature'] = 0.05
param_exp['dt_obs'] = 0.05
param_exp['dt_assim'] = 0.05
param_exp['spinup_length'] = 2000

param_bc = {}
param_bc['bc_type'] = 'linear'
param_bc['alpha'] = 0.01
param_bc['gamma'] = 0.0002
param_bc['offline'] = 'true'
#param_bc['offline'] = None
param_bc['path'] = param_exp['expdir'] + '/' + param_exp['obs_type'] + '/nocorr/coeffw.nc'
param_bc['correct_step'] = 'true'

param_add = {}
param_add['amp_add_bias']=8.0
param_add['amp_add_bias_2']=4.0
param_add['bias_mode'] = 'step'
