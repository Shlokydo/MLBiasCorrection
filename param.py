###
### Common experiment parameter
###

param_model = {}
param_model['dimension'] = 16
param_model['dimension_coupled'] = 256
param_model['forcing'] = 16
param_model['dt'] = 0.005
param_model['dt_coupled'] = 0.0001  # dt/c
param_model['h'] = 1
param_model['b'] = 20
param_model['c'] = 50
param_model['amp_const_bias'] = 1


param_exp = {}
param_exp['expdir'] = './DATA/coupled_bc_test'
param_exp['obs_type'] = 'test_obs'
param_exp['da_type'] = 'test_da'
param_exp['dt_nature'] = 0.05
param_exp['dt_obs'] = 0.05
param_exp['dt_assim'] = 0.05
param_exp['spinup_length'] = 2000
param_exp['ensembles'] = 20
param_exp['exp_length'] = 2000

param_letkf = {}
param_letkf['obs_error'] = 0.1
param_letkf['localization_length_scale'] = 3
param_letkf['localization_length_cutoff'] = 4
param_letkf['inflation_factor'] = 2.8

param_bc = {}
param_bc['bc_type'] = None
param_bc['alpha'] = 0.01
param_bc['gamma'] = 0.0002

