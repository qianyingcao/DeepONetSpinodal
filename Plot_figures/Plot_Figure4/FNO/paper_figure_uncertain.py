import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import time
import math
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy import io
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats
            


def plot_curves():

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 15
    plt.rcParams['lines.markersize'] = 6

    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'

    dpi = 600

    os.makedirs("paper_figure", exist_ok=True)

    iepoch_plot = 2
    negative_number = -2
    case_name1 = 'onlycase1'
    case_name2 = 'onlycase2'
    case_name3 = 'onlycase3'
    case_name4 = 'onlycase4'
    case_name5 = 'onlycase5'
    case_name6 = 'onlycase6'
    stress_normalizer = 50

    data_final_1 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name1, 'final'), allow_pickle=True).item()
    data_final_2 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name2, 'final'), allow_pickle=True).item()
    data_final_3 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name3, 'final'), allow_pickle=True).item()
    data_final_4 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name4, 'final'), allow_pickle=True).item()
    data_final_5 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name5, 'final'), allow_pickle=True).item()
    data_final_6 = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name6, 'final'), allow_pickle=True).item()
    #print(data_iepoch.keys())
    #print(data_final.keys())

    strain, stress_test_1, stress_test_pred_1, stress_weight_test_1, angle_test = data_final_1['data']
    strain, stress_test_2, stress_test_pred_2, stress_weight_test_2, angle_test = data_final_2['data']
    strain, stress_test_3, stress_test_pred_3, stress_weight_test_3, angle_test = data_final_3['data']
    strain, stress_test_4, stress_test_pred_4, stress_weight_test_4, angle_test = data_final_4['data']
    strain, stress_test_5, stress_test_pred_5, stress_weight_test_5, angle_test = data_final_5['data']
    strain, stress_test_6, stress_test_pred_6, stress_weight_test_6, angle_test = data_final_6['data']
    stress_test_1 = np.where(stress_test_1==negative_number, np.nan, stress_test_1)
    stress_test_pred_1 = np.where(stress_weight_test_1==0.0, np.nan, stress_test_pred_1)
    stress_test_1 *= stress_normalizer
    stress_test_pred_1 *= stress_normalizer
    stress_test_2 = np.where(stress_test_2==negative_number, np.nan, stress_test_2)
    stress_test_pred_2 = np.where(stress_weight_test_2==0.0, np.nan, stress_test_pred_2)
    stress_test_2 *= stress_normalizer
    stress_test_pred_2 *= stress_normalizer
    stress_test_3 = np.where(stress_test_3==negative_number, np.nan, stress_test_3)
    stress_test_pred_3 = np.where(stress_weight_test_3==0.0, np.nan, stress_test_pred_3)
    stress_test_3 *= stress_normalizer
    stress_test_pred_3 *= stress_normalizer
    stress_test_4 = np.where(stress_test_4==negative_number, np.nan, stress_test_4)
    stress_test_pred_4 = np.where(stress_weight_test_4==0.0, np.nan, stress_test_pred_4)
    stress_test_4 *= stress_normalizer
    stress_test_pred_4 *= stress_normalizer
    stress_test_5 = np.where(stress_test_5==negative_number, np.nan, stress_test_5)
    stress_test_pred_5 = np.where(stress_weight_test_5==0.0, np.nan, stress_test_pred_5)
    stress_test_5 *= stress_normalizer
    stress_test_pred_5 *= stress_normalizer
    stress_test_6 = np.where(stress_test_6==negative_number, np.nan, stress_test_6)
    stress_test_pred_6 = np.where(stress_weight_test_6==0.0, np.nan, stress_test_pred_6)
    stress_test_6 *= stress_normalizer
    stress_test_pred_6 *= stress_normalizer


    stress_test = np.concatenate([stress_test_1[None,:], stress_test_2[None,:], stress_test_3[None,:], stress_test_4[None,:], stress_test_5[None,:], stress_test_6[None,:]], axis=0)
    stress_test_pred = np.concatenate([stress_test_pred_1[None,:], stress_test_pred_2[None,:], stress_test_pred_3[None,:], stress_test_pred_4[None,:], stress_test_pred_5[None,:], stress_test_pred_6[None,:]], axis=0)
    stress_test_mean = np.nanmean(stress_test, axis=0)
    stress_test_pred_mean = np.nanmean(stress_test_pred, axis=0)
    stress_test_std = np.nanstd(stress_test, axis=0)
    stress_test_pred_std = np.nanstd(stress_test_pred, axis=0)
    

    ### Stress-Strain Curve Comparison ###
    plot_idx_list = [136, 122, 132, 33, 10]
    num_samples = len(plot_idx_list)
    fig, axs = plt.subplots(nrows = 3, ncols = num_samples, figsize = (3.2*num_samples,2.4*3), dpi=dpi)
    L2_err_all = []
    L2_relerr_all = []
    for iplot, idx_plot in enumerate(plot_idx_list):
        def single_subplot(y, y_pred, y_test_std, y_std, irow, icol, title):
            color_map = {0:'#1f77b4',1:'#ff7f0e',2:'#2ca02c',3:'c',4:'m',5:'k'}
            strain_plot = strain            
            axs[irow,icol].plot(strain_plot, y[0], linestyle='--', color=color_map[irow+3])
            axs[irow,icol].plot(strain_plot, y[1], linestyle='--', color=color_map[irow+3])            
            axs[irow,icol].fill_between(strain_plot, y[0] + 2*y_test_std[0], y[0] - 2*y_test_std[0], alpha=0.3, color=color_map[irow+3])
            axs[irow,icol].fill_between(strain_plot, y[1] + 2*y_test_std[1], y[1] - 2*y_test_std[1], alpha=0.3, color=color_map[irow+3])
            axs[irow,icol].plot(strain_plot, y_pred[0],  linestyle='-', color=color_map[irow])
            axs[irow,icol].plot(strain_plot, y_pred[1],  linestyle='-', color=color_map[irow])
            axs[irow,icol].fill_between(strain_plot, y_pred[0] + 2*y_std[0], y_pred[0] - 2*y_std[0], alpha=0.3, color=color_map[irow])
            axs[irow,icol].fill_between(strain_plot, y_pred[1] + 2*y_std[1], y_pred[1] - 2*y_std[1], alpha=0.3, color=color_map[irow])
            axs[irow, icol].tick_params(axis='x', labelsize=14)
            axs[irow, icol].tick_params(axis='y', labelsize=14)
            axs[irow, icol].set_yticklabels([])
            axs[irow, icol].set_yticks([])
            if icol == 0 or icol == 3:
                axs[irow,icol].set_ylim([0,85])
            else:
                axs[irow,icol].set_ylim([0,70])


            if icol == 0 or icol == 1 or icol == 2 or icol == 3:
                axs[irow,icol].set_xticklabels([])
                axs[irow, icol].set_xticks([])

            #axs[irow,icol].set_xlabel('Strain')
            #axs[irow,icol].set_ylabel('Stress (MPa)')
            return
        stress_test_plot = stress_test_mean[idx_plot]
        stress_pred_plot = stress_test_pred_mean[idx_plot]
        stress_test_std_plot = stress_test_std[idx_plot]
        stress_pred_std_plot = stress_test_pred_std[idx_plot]

        single_subplot(stress_test_plot[0], stress_pred_plot[0], stress_test_std_plot[0], stress_pred_std_plot[0], 0, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))
        single_subplot(stress_test_plot[1], stress_pred_plot[1], stress_test_std_plot[1], stress_pred_std_plot[1], 1, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))
        single_subplot(stress_test_plot[2], stress_pred_plot[2], stress_test_std_plot[2], stress_pred_std_plot[2], 2, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))
        diff = stress_pred_plot - stress_test_plot
        print(stress_pred_plot.shape)
        print(stress_test_plot.shape)
        L2_err = np.sqrt(np.nanmean(diff**2))
        L2_relerr = np.sqrt(np.nanmean(diff**2)) / np.sqrt(np.nanmean(stress_test_plot**2))
        L2_err_all.append(L2_err)
        L2_relerr_all.append(L2_relerr)

        # saved_mat = {'strain':strain, 'stress_target':stress_test_plot, 'stress_pred':stress_pred_plot}
        # io.savemat('./{}/Output/curve_{}.mat'.format(CASE_NAME, iplot+1), saved_mat)
    #plt.show()

    L2_err_mean = np.mean(L2_err_all)
    L2_relerr_mean = np.mean(L2_relerr_all)

    print('L2 Rel. Err.')
    print(L2_relerr_all)


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('./Curve_uncertainty.png')
    plt.close()


if __name__ == '__main__':
    plot_curves()
