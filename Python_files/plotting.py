from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
import helperfunctions as helpfunc
import os

import mlflow
mlflow.set_tracking_uri('file:/home/mshlok/MLBiasCorrection/Python_files/mlruns_plot')
print(mlflow.tracking.get_tracking_uri())

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 7,
        }

def scatter_plot(plot_variable, variable_num, x1_label, x2_label, y_label, directory, c_rmse, rmse):

    x1, x2, y = plot_variable
    fig, ax = plt.subplots()
    fig.suptitle('Variable {}'.format(variable_num))

    ax.set_xlabel(y_label)
    ax.set_ylabel(y_label)
    ax.plot(y, y, c = 'r', linewidth = 1)
    ax.scatter(y, x2, s=10, marker='o', c = 'b', label= x2_label)
    ax.scatter(y, x1, s=8, marker='*', c = 'g', label=x1_label)
    ax.text(-1.1, 4, 'Corrected RMSE: ' + str(c_rmse), fontdict = font)
    ax.text(-1.1, 3.7, 'Biased RMSE: ' + str(rmse), fontdict = font)
    ax.legend()

    img_name = directory + '/scatter_plot_variable_{}.png'.format(variable_num)
    print('Saving image file: {}'.format(img_name))
    fig.savefig(img_name, format= 'png', dpi = 1200)

def line_plot(plot_variable, variable_num, directory, c_rmse, rmse):

    model_forecast, forecast,  analysis = plot_variable
    time = np.arange(len(analysis)) + 1
    
    fig, ax = plt.subplots()
    ax.set_title('Variable {}'.format(variable_num))
    ax.set_xlabel('Time')
    ax.plot(time, analysis, 'g-', linewidth = 2 , label = 'Analysis')
    ax.plot(time, forecast, 'y--',  linewidth = 1, label = 'Forecast')
    ax.plot(time, model_forecast, 'k:', linewidth = 1, label = 'Corrected forecast')
    ax.text(0, 4, 'Corrected RMSE: ' + str(c_rmse), fontdict = font)
    ax.text(0, 3.7, 'Biased RMSE: ' + str(rmse), fontdict = font)
    ax.legend()
    img_name = directory + '/line_plot_variable_{}.png'.format(variable_num)
    print('Saving image file: {}'.format(img_name))
    fig.savefig(img_name, format= 'png', dpi = 1200)

def plot_func(plist, c_forecast, analysis, forecast, c_rmse, rmse):
    mlflow.set_experiment('Plots')

    with mlflow.start_run(run_name = plist['experiment_name']):
        #Randomly select five variables for plotting
        random_variables = np.random.randint(low = 0, high=analysis.shape[1], size=5)
        image_dir = (plist['experiment_dir'] + '/images')
        if not(os.path.exists(image_dir)):
            os.mkdir(image_dir)

        for i in random_variables:
            
            scatter_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, 'Corrected_forecast', 'Biased_Forecast', 'Analysis', image_dir, c_rmse, rmse)
            
            line_plot((c_forecast[:,i], forecast[:,i], analysis[:,i]), i, image_dir, c_rmse, rmse)

        mlflow.log_artifacts(image_dir)
