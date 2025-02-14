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
    #num_samples = 20
    #sample_idx = np.random.permutation(num_test)[:num_samples]
    #plot_idx_list = [12,1,6,14,86,0]
    #plot_idx_list = [7, 16, 0, 170, 37, 33]
    plot_idx_list = [136, 122, 132, 33, 10]
    num_samples = len(plot_idx_list)
    fig, axs = plt.subplots(nrows = 3, ncols = num_samples, figsize = (3.2*num_samples,2.4*3), dpi=dpi)
    L2_err_all = []
    L2_relerr_all = []
    for iplot, idx_plot in enumerate(plot_idx_list):
        def single_subplot(y, y_pred, y_test_std, y_std, irow, icol, title):
            color_map = {0:'#1f77b4',1:'#ff7f0e',2:'#2ca02c',3:'c',4:'m',5:'k'}
            strain_plot = strain            
            axs[irow,icol].plot(strain_plot, y[0], linestyle='--', color=color_map[irow+3], label='Exp')
            axs[irow,icol].plot(strain_plot, y[1], linestyle='--', color=color_map[irow+3])            
            axs[irow,icol].fill_between(strain_plot, y[0] + 2*y_test_std[0], y[0] - 2*y_test_std[0], alpha=0.3, color=color_map[irow+3])
            axs[irow,icol].fill_between(strain_plot, y[1] + 2*y_test_std[1], y[1] - 2*y_test_std[1], alpha=0.3, color=color_map[irow+3])
            axs[irow,icol].plot(strain_plot, y_pred[0],  linestyle='-', color=color_map[irow], label='Pred')
            axs[irow,icol].plot(strain_plot, y_pred[1],  linestyle='-', color=color_map[irow])
            axs[irow,icol].fill_between(strain_plot, y_pred[0] + 2*y_std[0], y_pred[0] - 2*y_std[0], alpha=0.3, color=color_map[irow])
            axs[irow,icol].fill_between(strain_plot, y_pred[1] + 2*y_std[1], y_pred[1] - 2*y_std[1], alpha=0.3, color=color_map[irow])
            axs[irow, icol].tick_params(axis='x', labelsize=14)
            axs[irow, icol].tick_params(axis='y', labelsize=14)
            if icol == 0 or icol == 3:
                axs[irow,icol].set_ylim([0,85])
            else:
                axs[irow,icol].set_ylim([0,70])

            if irow == 1 or irow == 2:
                axs[irow,icol].set_yticklabels([])
                axs[irow, icol].set_yticks([])

            if icol == 0 or icol == 1 or icol == 2 or icol == 3:
                axs[irow,icol].set_xticklabels([])
                axs[irow, icol].set_xticks([])

            if irow == 0 and icol == 0:
                axs[irow,icol].legend(loc='upper right')
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

    # def calc_r2(pred, true):
    #     pred = pred.flatten()
    #     true = true.flatten()


    # def calc_relative_err(pred, true):
    #     # TODO: eliminate the axis 0
    #     numerator = np.sqrt(np.nanmean((pred-true)**2))
    #     denominator = np.sqrt(np.nanmean((true)**2))
    #     ans = numerator / denominator
    #     return ans

    # # Mechanical Parameters
    # def calc_energy_per(stress):
    #     loading_energy = np.nansum(stress[:,:,0],axis=2)
    #     unloading_energy = np.nansum(stress[:,:,1],axis=2)
    #     dissipation = 1 - unloading_energy/loading_energy
    #     return dissipation*100
    # def calc_energy_abs(stress):
    #     loading_energy = np.nansum(stress[:,:,0],axis=2) * (strain[1] - strain[0])
    #     unloading_energy = np.nansum(stress[:,:,1],axis=2) * (strain[1] - strain[0])
    #     dissipation = loading_energy - unloading_energy
    #     return dissipation
    # dissipation_test_qr = calc_energy_abs(stress_test_qr)
    # dissipation_test_pred_qr = calc_energy_abs(stress_test_pred_qr)
    # dissipation_test_fno = calc_energy_abs(stress_test_fno)
    # dissipation_test_pred_fno = calc_energy_abs(stress_test_pred_fno)
    # #dissipation_relerr = calc_relative_err(dissipation_test_pred, dissipation_test)

    # def calc_max_stress(stress):
    #     stress_max = np.nanmax(stress[:,:,0],axis=2)
    #     return stress_max
    # stress_max_test_qr = calc_max_stress(stress_test_qr)
    # stress_max_test_pred_qr = calc_max_stress(stress_test_pred_qr)
    # stress_max_test_fno = calc_max_stress(stress_test_fno)
    # stress_max_test_pred_fno = calc_max_stress(stress_test_pred_fno)
    # #stress_max_relerr = calc_relative_err(stress_max_test_pred_qr, stress_max_test_qr)

    # # Create a new figure with two subplots
    # fig, axs = plt.subplots(1, 4, figsize=(3.6*4,3.6), dpi=dpi)

    # # All Stress
    # ax = axs[0]
    # alpha = 0.01
    # color_map = ['r','r','r']
    # ax.scatter(stress_test_qr[:,0].flatten(), stress_test_pred_qr[:,0].flatten(), c=color_map[0], alpha=alpha, s=3)
    # ax.scatter(stress_test_qr[:,1].flatten(), stress_test_pred_qr[:,1].flatten(), c=color_map[1], alpha=alpha, s=3)
    # ax.scatter(stress_test_qr[:,2].flatten(), stress_test_pred_qr[:,2].flatten(), c=color_map[2], alpha=alpha, s=3)
    # ax.set_title("All Stress (MPa)")
    # ax.set_xlabel("Exp", color='red')
    # ax.set_ylabel("Pred (DeepONet)", color='red')
    # ax.set_xlim(0,100)
    # ax.set_ylim(0,100)
    # ax.set_xticks([0,25,50,75,100])
    # ax.set_yticks([0,25,50,75,100])
    # ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    # ax.tick_params(direction="in",which='both')
    # ax.tick_params('y', colors='red')
    # ax.tick_params('x', colors='red')

    # not_nan = np.logical_and(np.logical_not(np.isnan(stress_test_qr.flatten())), np.logical_not(np.isnan(stress_test_pred_qr.flatten())))
    # _, _, r_value, _, _ = stats.linregress(stress_test_qr.flatten()[not_nan], stress_test_pred_qr.flatten()[not_nan])
    # print('All Stress R2: {}'.format(r_value**2))

    # ax2 = ax.twinx()
    # ax3 = ax.twiny()

    # ax2.scatter(stress_test_fno[:,0].flatten(), stress_test_pred_fno[:,0].flatten(), c='b', alpha=alpha, s=3)
    # ax2.scatter(stress_test_fno[:,1].flatten(), stress_test_pred_fno[:,1].flatten(), c='b', alpha=alpha, s=3)
    # ax2.scatter(stress_test_fno[:,2].flatten(), stress_test_pred_fno[:,2].flatten(), c='b', alpha=alpha, s=3)
    # ax3.set_xticks([0,25,50,75,100])
    # ax3.set_xlabel("Exp", color='blue')
    # #ax2.set_ylabel("Pred (FNO)")
    # ax3.set_aspect('equal')
    # ax3.set_box_aspect(1)
    # ax2.tick_params('y', colors='blue')
    # ax3.tick_params('x', colors='blue')

    # # Inverting the axis limits for FNO to form the 'X' shape
    # ax2.set_ylim(ax.get_ylim()[::-1])
    # ax3.plot(ax3.get_xlim(), ax2.get_ylim(), ls="--", c="black")
    # ax3.set_aspect('equal')


    # not_nan = np.logical_and(np.logical_not(np.isnan(stress_test_fno.flatten())), np.logical_not(np.isnan(stress_test_pred_fno.flatten())))
    # _, _, r_value, _, _ = stats.linregress(stress_test_fno.flatten()[not_nan], stress_test_pred_fno.flatten()[not_nan])
    # print('All Stress FNO R2: {}'.format(r_value**2))


    # what_plot = 2           # 1: scatter; 2: bar
    # def group_points(x, y):
    #     # values of x are limited, group by x
    #     x_group, y_group, y_err_group = np.zeros((num_angles,)), np.zeros((num_angles,)), np.zeros((num_angles,))
    #     for iangle in range(num_angles):
    #         cur_angle = angle_unique_list[iangle]
    #         is_this_angle = np.array([angle == cur_angle for angle in angle_list])
    #         x_group[iangle] = x[is_this_angle][0]
    #         y_group[iangle] = y[is_this_angle].mean()
    #         y_err_group[iangle] = y[is_this_angle].std()
    #     return x_group, y_group, y_err_group

    # # Energy Absorption
    # ax = axs[1]
    # alpha = 0.7
    # color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    # if what_plot == 1:
    #     ax.scatter(dissipation_test_qr[:,0], dissipation_test_pred_qr[:,0], c=color_map[0], alpha=alpha)
    #     ax.scatter(dissipation_test_qr[:,1], dissipation_test_pred_qr[:,1], c=color_map[1], alpha=alpha)
    #     ax.scatter(dissipation_test_qr[:,2], dissipation_test_pred_qr[:,2], c=color_map[2], alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x, y, yerr = group_points(dissipation_test_qr[:,idir], dissipation_test_pred_qr[:,idir])
    #         ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='r', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax.set_title(r"Energy Absorption (MPa)")
    # ax.set_xlabel("Exp", color='red')
    # #ax.set_ylabel("Pred (DeepONet)")
    # ax.set_xlim(-0.2,10.2)
    # ax.set_ylim(-0.2,10.2)
    # ax.set_xticks([0,2.5,5,7.5,10])
    # ax.set_yticks([0,2.5,5,7.5,10])
    # ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    # ax.tick_params(direction="in",which='both')
    # ax.tick_params('y', colors='red')
    # ax.tick_params('x', colors='red')
    # _, _, r_value_qr, _, _ = stats.linregress(dissipation_test_qr.flatten(), dissipation_test_pred_qr.flatten())
    # print('All Energy Absorption R2: {}'.format(r_value_qr**2))
    # if what_plot == 2:
    #     _, _, r_value_qr, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Energy Absorption R2: {}'.format(r_value_qr**2))

    # ax2 = ax.twinx()
    # ax3 = ax.twiny()
    # if what_plot == 1:
    #     ax2.scatter(dissipation_test_fno[:,0], dissipation_test_pred_fno[:,0], c=color_map[0], alpha=alpha)
    #     ax2.scatter(dissipation_test_fno[:,1], dissipation_test_pred_fno[:,1], c=color_map[1], alpha=alpha)
    #     ax2.scatter(dissipation_test_fno[:,2], dissipation_test_pred_fno[:,2], c=color_map[2], alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x, y, yerr = group_points(dissipation_test_fno[:,idir], dissipation_test_pred_fno[:,idir])
    #         ax2.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='b', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax3.set_xticks([0,2.5,5,7.5,10])
    # ax2.set_yticks([0,2.5,5,7.5,10])
    # ax3.set_xlabel("Exp", color='blue')
    # #ax2.set_ylabel("Pred (FNO)")
    # ax3.set_aspect('equal')
    # ax3.set_box_aspect(1)
    # ax2.tick_params('y', colors='blue')
    # ax3.tick_params('x', colors='blue')

    # # Inverting the axis limits for FNO to form the 'X' shape
    # ax2.set_ylim(ax.get_ylim()[::-1])
    # ax3.plot(ax3.get_xlim(), ax2.get_ylim(), ls="--", c="black")
    # ax3.set_aspect('equal')
    
    # _, _, r_value_fno, _, _ = stats.linregress(dissipation_test_fno.flatten(), dissipation_test_pred_fno.flatten())
    # print('All Energy Absorption FNO R2: {}'.format(r_value_fno**2))
    # if what_plot == 2:
    #     _, _, r_value_fno, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Energy Absorption FNO R2: {}'.format(r_value_fno**2))



    # # Max Stress
    # ax = axs[2]
    # alpha = 0.7
    # color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    # if what_plot == 1:
    #     ax.scatter(stress_max_test_qr[:,0], stress_max_test_pred_qr[:,0], c='r', alpha=alpha)
    #     ax.scatter(stress_max_test_qr[:,1], stress_max_test_pred_qr[:,1], c='r', alpha=alpha)
    #     ax.scatter(stress_max_test_qr[:,2], stress_max_test_pred_qr[:,2], c='r', alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x, y, yerr = group_points(stress_max_test_qr[:,idir], stress_max_test_pred_qr[:,idir])
    #         ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='r', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax.set_title("Ultimate Stress (MPa)")
    # ax.set_xlabel("Exp", color='red')
    # #ax.set_ylabel("Pred (DeepONet)")
    # ax.set_xlim(-2,102)
    # ax.set_ylim(-2,102)
    # ax.set_xticks([0,25,50,75,100])
    # ax.set_yticks([0,25,50,75,100])
    # ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    # ax.tick_params(direction="in",which='both')
    # ax.tick_params('y', colors='red')
    # ax.tick_params('x', colors='red')
    # _, _, r_value, _, _ = stats.linregress(stress_max_test_qr.flatten(), stress_max_test_pred_qr.flatten())
    # print('All Max Stress R2: {}'.format(r_value**2))
    # if what_plot == 2:
    #     _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Max Stress R2: {}'.format(r_value**2))

    # ax2 = ax.twinx()
    # ax3 = ax.twiny()
    # if what_plot == 1:
    #     ax2.scatter(stress_max_test_fno[:,0], stress_max_test_pred_fno[:,0], c='b', alpha=alpha)
    #     ax2.scatter(stress_max_test_fno[:,1], stress_max_test_pred_fno[:,1], c='b', alpha=alpha)
    #     ax2.scatter(stress_max_test_fno[:,2], stress_max_test_pred_fno[:,2], c='b', alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x, y, yerr = group_points(stress_max_test_fno[:,idir], stress_max_test_pred_fno[:,idir])
    #         ax2.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='b', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax3.set_xticks([0,25,50,75,100])
    # ax2.set_yticks([0,25,50,75,100])
    # ax3.set_xlabel("Exp", color='blue')
    # #ax2.set_ylabel("Pred (FNO)")
    # ax3.set_aspect('equal')
    # ax3.set_box_aspect(1)
    # ax2.tick_params('y', colors='blue')
    # ax3.tick_params('x', colors='blue')

    # # Inverting the axis limits for FNO to form the 'X' shape
    # ax2.set_ylim(ax.get_ylim()[::-1])
    # ax3.plot(ax3.get_xlim(), ax2.get_ylim(), ls="--", c="black")
    # ax3.set_aspect('equal')

    # _, _, r_value, _, _ = stats.linregress(stress_max_test_fno.flatten(), stress_max_test_pred_fno.flatten())
    # print('All Max Stress FNO R2: {}'.format(r_value**2))
    # if what_plot == 2:
    #     _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Max Stress FNO R2: {}'.format(r_value**2))
    

    # # Stiffness
    # def linear_fit(x, y):
    #     x = np.tile(x, (y.shape[0], 1))
    #     x_mean = np.mean(x,axis=1,keepdims=True)
    #     y_mean = np.mean(y,axis=1,keepdims=True)
    #     x_bar = x - x_mean
    #     y_bar = y - y_mean
    #     slope = np.sum(y_bar * x_bar, axis=1) / np.sum(x_bar * x_bar, axis=1)
    #     return slope
    # ax = axs[3]
    # alpha = 0.7
    # color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    # num_pts = 5
    # #print(stress_test[:,0,0,1]/strain[1])
    # if what_plot == 1:
    #     ax.scatter(linear_fit(strain[:num_pts], stress_test_qr[:,0,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_qr[:,0,0,:num_pts]), c=color_map[0], alpha=alpha)
    #     ax.scatter(linear_fit(strain[:num_pts], stress_test_qr[:,1,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_qr[:,1,0,:num_pts]), c=color_map[1], alpha=alpha)
    #     ax.scatter(linear_fit(strain[:num_pts], stress_test_qr[:,2,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_qr[:,2,0,:num_pts]), c=color_map[2], alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x0, y0 = linear_fit(strain[:num_pts], stress_test_qr[:,idir,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_qr[:,idir,0,:num_pts]),
    #         x, y, yerr = group_points(x0, y0)
    #         ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='r', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax.set_title("Stiffness (MPa)")
    # ax.set_xlabel("Exp", color='red')
    # #ax.set_ylabel("Pred (DeepONet)")
    # ax.set_xlim(194,506)
    # ax.set_ylim(194,506)
    # ax.set_xticks([200,300,400,500])
    # ax.set_yticks([200,300,400,500])
    # ax.set_aspect('equal')
    # ax.set_box_aspect(1)
    # ax.tick_params(direction="in",which='both')
    # ax.tick_params('y', colors='red')
    # ax.tick_params('x', colors='red')
    # stiffness_test = np.concatenate([linear_fit(strain[:num_pts], stress_test_qr[:,i,0,:num_pts]) for i in range(3)], axis=0)
    # stiffness_test_pred = np.concatenate([linear_fit(strain[:num_pts], stress_test_pred_qr[:,i,0,:num_pts]) for i in range(3)], axis=0)
    # _, _, r_value, _, _ = stats.linregress(stiffness_test.flatten(), stiffness_test_pred.flatten())
    # print('All Stiffness R2: {}'.format(r_value**2))
    # if what_plot == 2:
    #     _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Stiffness R2: {}'.format(r_value**2))

    # ax2 = ax.twinx()
    # ax3 = ax.twiny()
    # if what_plot == 1:
    #     ax2.scatter(linear_fit(strain[:num_pts], stress_test_fno[:,0,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_fno[:,0,0,:num_pts]), c='b', alpha=alpha)
    #     ax2.scatter(linear_fit(strain[:num_pts], stress_test_fno[:,1,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_fno[:,1,0,:num_pts]), c='b', alpha=alpha)
    #     ax2.scatter(linear_fit(strain[:num_pts], stress_test_fno[:,2,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_fno[:,2,0,:num_pts]), c='b', alpha=alpha)
    # else:
    #     x_all, y_all = [], []
    #     for idir in range(3):
    #         x0, y0 = linear_fit(strain[:num_pts], stress_test_fno[:,idir,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred_fno[:,idir,0,:num_pts]),
    #         x, y, yerr = group_points(x0, y0)
    #         ax2.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c='b', alpha=alpha)
    #         x_all.append(x)
    #         y_all.append(y)
    #     x_all = np.concatenate(x_all, axis=0)
    #     y_all = np.concatenate(y_all, axis=0)
    # ax3.set_xticks([200,300,400,500])
    # ax2.set_yticks([200,300,400,500])
    # ax3.set_xlabel("Exp", color='blue')
    # ax2.set_ylabel("Pred (FNO)", color='blue')
    # ax3.set_aspect('equal')
    # ax3.set_box_aspect(1)
    # ax2.tick_params('y', colors='blue')
    # ax3.tick_params('x', colors='blue')

    # # Inverting the axis limits for FNO to form the 'X' shape
    # ax2.set_ylim(ax.get_ylim()[::-1])
    # ax3.plot(ax3.get_xlim(), ax2.get_ylim(), ls="--", c="black")
    # ax3.set_aspect('equal')

    # stiffness_test = np.concatenate([linear_fit(strain[:num_pts], stress_test_fno[:,i,0,:num_pts]) for i in range(3)], axis=0)
    # stiffness_test_pred = np.concatenate([linear_fit(strain[:num_pts], stress_test_pred_fno[:,i,0,:num_pts]) for i in range(3)], axis=0)
    # _, _, r_value, _, _ = stats.linregress(stiffness_test.flatten(), stiffness_test_pred.flatten())
    # print('All Stiffness FNO R2: {}'.format(r_value**2))
    # if what_plot == 2:
    #     _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
    #     print('Mean Stiffness FNO R2: {}'.format(r_value**2))

    


    # # Plot dashed diagonal lines for both subplots
    # for ax in axs:
    #     ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black")
    #     ax.set_aspect('equal')

    # # Adjust spacing and display the plot
    # plt.tight_layout()
    # plt.savefig('./paper_figure/Fig_forward_parameters.pdf')
    # plt.close()


if __name__ == '__main__':
    plot_curves()
