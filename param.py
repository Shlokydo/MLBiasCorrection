###
### Common experiment parameter
###

param_model = {}
param_model['dimension'] = 40
param_model['dimension_coupled'] = 1600
param_model['forcing'] = 8
param_model['dt'] = 0.005
param_model['dt_coupled'] = 0.0005  # dt/c
param_model['h'] = 1
param_model['b'] = 10
param_model['c'] = 10
param_model['amp_const_bias'] = 1


param_exp = {}
param_exp['expdir'] = './DATA/coupled'
param_exp['obs_type'] = 'test_obs'
param_exp['da_type'] = 'test_da'
param_exp['dt_nature'] = 0.05
param_exp['dt_obs'] = 0.2
param_exp['dt_assim'] = 0.2
param_exp['spinup_length'] = 200
param_exp['ensembles'] = 20
param_exp['exp_length'] = 200

param_letkf = {}
param_letkf['obs_error'] = 1.0
param_letkf['localization_length_scale'] = 3
param_letkf['localization_length_cutoff'] = 9
param_letkf['inflation_factor'] = 2.0

