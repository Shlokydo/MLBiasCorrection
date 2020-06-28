#------------------------------------------------
import os
import sys
import numpy as np
import param
import matplotlib.pyplot as plt

#------------------------------------------------
### config
nx  = param.param_model['dimension'] 

#exp = 'coupled'
#exp = 'sint_x2'
exp = 'coupled_020'

#items = ('nocorr','linear','D_5','L2D_5','D_3','L2D_3')
items = ('nocorr','linear','linear4','D_5','L2D_5')
#items = ('nocorr','CLD_5_5','CLD_7_5','CLD_10_5')
#legends = ('nocorr','LSTM(5)','LSTM(7)','LSTM(10)')
legends = ('nocorr','linear','4th poly','Dense-5','LSTM-5')
#legends = ('nocorr','linear','Dense-5','LSTM-5','Dense-3','LSTM-3')
lstyles = ('solid','dotted',(0,(1,4)),'dashed','dashdot','dashed','dashdot')

#------------------------------------------------

nitem=len(items)

infl=[]
rmse=[]

# load nature and observation data
for i in range(nitem):
  infl.append([])
  rmse.append([])
  fname='./result_txt/log_sweep_assim_' + exp + '_' + items[i] 
  with open(fname,'r') as f:
    line=f.readlines()
    length=len(line)
    for j in range(length): 
      infl[i].append(float(line[j][13:16]))
      rmse[i].append(float(line[j][29:40]))

###
plt.figure()
plt.yscale('log')
#plt.scatter(test_labels, test_predictions)
#plt.scatter(fcst[ntime-100:ntime,1], anal[ntime-100:ntime,1]-fcst[ntime-100:ntime,1])
for i in range(nitem):
  plt.plot(infl[i], rmse[i],label=legends[i],linestyle=lstyles[i])
# plt.plot(time, sprd_plot)

plt.legend(bbox_to_anchor=(0.80,0.98), loc='upper right', borderaxespad=0,fontsize=14)
#plt.plot(refx, refy,color='black',linestyle='dashed')
plt.xlabel('Inflation factor')
plt.ylabel('Analysis RMSE')
#plt.axis('equal')
#plt.axis('square')
plt.xlim()
plt.ylim()
#plt.xlim([0,plt.xlim()[1]])
#plt.ylim([0,plt.ylim()[1]])
#_ = plt.plot([-100, 100], [-100, 100])
plt.savefig('infl.png')
###

