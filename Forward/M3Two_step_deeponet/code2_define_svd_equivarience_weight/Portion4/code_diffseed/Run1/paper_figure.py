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
    case_name = 'onlycase'
    stress_normalizer = 50

    data_final = np.load('./{}/Output/SavedOutputs_{}.npy'.format(case_name, 'final'), allow_pickle=True).item()
    #print(data_iepoch.keys())
    #print(data_final.keys())

    strain, stress_test, stress_test_pred, stress_weight_test, angle_test = data_final['data']
    stress_test = np.where(stress_test==negative_number, np.nan, stress_test)
    stress_test *= stress_normalizer
    stress_test_pred *= stress_normalizer

    angle_list = []
    angle_unique_set = set()
    for angle in angle_test:
        angle_unique_set.add(tuple(angle))
        angle_list.append(tuple(angle))
    angle_unique_list = list(angle_unique_set)
    num_angles = len(angle_unique_set)
    angle_to_id = {angle_unique_list[iangle]: iangle for iangle in range(num_angles)}
    id_to_angle = {iangle: angle_unique_list[iangle] for iangle in range(num_angles)}

    num_test = stress_test_pred.shape[0]

    num_stages = 2

    ### Stress-Strain Curve Comparison ###
    #num_samples = 20
    #sample_idx = np.random.permutation(num_test)[:num_samples]
    #plot_idx_list = [3,7,13,12,18]
    plot_idx_list = [3, 18, 21, 4, 15,6]
    num_samples = len(plot_idx_list)
    fig, axs = plt.subplots(nrows = 3, ncols = num_samples, figsize = (3.2*num_samples,2.4*3), dpi=dpi)
    for iplot, idx_plot in enumerate(plot_idx_list):

        def extend_right(x, y):
            not_nan = np.logical_not(np.isnan(y[0,:]))
            x = x[not_nan]
            y = y[:,not_nan]
            x_1, x_2 = x[-2], x[-1]
            y1_1, y1_2 = y[0,-2], y[0,-1]
            y2_1, y2_2 = y[1,-2], y[1,-1]
            relative_slope = (y2_2 - y2_1 - y1_2 + y1_1) / (x_2 - x_1)
            dy = y1_2 - y2_2
            dx = dy / relative_slope
            x_new = x_2 + dx
            y_new = y1_2 + dx * (y1_2 - y1_1) / (x_2 - x_1)
            x_extended = np.hstack((x,[x_new]))
            y_extended = np.concatenate((y, np.array([y_new, y_new]).reshape([-1,1])), axis=1)
            return x_extended, y_extended

        def single_subplot(y, y_pred, irow, icol, title):
            ax = axs[irow,icol]
            color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
            strain_pred_extended, y_pred_extended = extend_right(strain, y_pred)
            strain_extended, y_extended = extend_right(strain, y)
            istage = 0
            ax.plot(strain_pred_extended, y_pred_extended[istage,:], linestyle='--',color=color_map[irow])
            ax.plot(strain_extended, y_extended[istage,:], linestyle='-',color=color_map[irow])
            istage = 1
            #unload_zero_crop = lambda x: np.where(x>=0, x, np.where(x<0, 0, np.nan))
            ax.plot(strain_pred_extended, y_pred_extended[istage,:], linestyle='--',color=color_map[irow])
            ax.plot(strain_extended, y_extended[istage,:], linestyle='-',color=color_map[irow])
            if irow == 2 and icol == 2: ax.set_xlabel(r'$\varepsilon$',fontsize=20)
            if irow == 1 and icol == 0: ax.set_ylabel(r'$\sigma$', rotation=0, labelpad=15, fontsize=20)
            ax.set_xticks([0.0,0.1,0.2,0.3])
            #y_max = 1.2*max(np.nanmax(y),np.nanmax(y_pred))
            #ax.set_yticks([0.0,0.5*y_max,y_max])
            if irow == 0: ax.set_title(title, pad=10)
            ax.tick_params(direction="in",which='both')
            ax.set_xticks([0,0.1,0.2,0.3])
            ax.set_yticks([0,50,100])
            ax.set_xlim([-0.006, 0.306])
            ax.set_ylim([-5,105])
            return
        
        def plot_process(stress):
            # stress: (3, num_stages, num_pts)
            stress_copy = np.copy(stress)
            for idir in range(3):
                for istage in range(num_stages):
                    if istage%2 == 1:
                        stress_i = stress_copy[idir,istage]
                        pointer = stress_i.shape[0]-1
                        while pointer>=0 and (not stress_i[pointer]<0):
                            pointer -= 1
                        pointer += 1
                        stress_copy[idir,istage,:pointer] = 0
            return stress_copy
        
        stress_test_pred_plot = plot_process(stress_test_pred[idx_plot])
        stress_test_plot = plot_process(stress_test[idx_plot])
        single_subplot(stress_test_plot[0], stress_test_pred_plot[0], 0, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))
        single_subplot(stress_test_plot[1], stress_test_pred_plot[1], 1, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))
        single_subplot(stress_test_plot[2], stress_test_pred_plot[2], 2, iplot, '({}, {}, {})'.format(*angle_test[idx_plot]))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.suptitle('Training Epoch #{} (L2 Err: {:.4f}; Rel Err: {:.4f})'.format(iepoch+1, L2_err_mean, L2_relerr_mean))
    plt.savefig('./paper_figure/Fig_forward_curve.pdf')
    plt.close()

    def calc_r2(pred, true):
        pred = pred.flatten()
        true = true.flatten()


    def calc_relative_err(pred, true):
        # TODO: eliminate the axis 0
        numerator = np.sqrt(np.nanmean((pred-true)**2))
        denominator = np.sqrt(np.nanmean((true)**2))
        ans = numerator / denominator
        return ans

    # Mechanical Parameters
    def calc_energy_per(stress):
        loading_energy = np.nansum(stress[:,:,0],axis=2)
        unloading_energy = np.nansum(stress[:,:,1],axis=2)
        dissipation = 1 - unloading_energy/loading_energy
        return dissipation*100
    def calc_energy_abs(stress):
        loading_energy = np.nansum(stress[:,:,0],axis=2) * (strain[1] - strain[0])
        unloading_energy = np.nansum(stress[:,:,1],axis=2) * (strain[1] - strain[0])
        dissipation = loading_energy - unloading_energy
        return dissipation
    dissipation_test = calc_energy_abs(stress_test)
    dissipation_test_pred = calc_energy_abs(stress_test_pred)
    #dissipation_relerr = calc_relative_err(dissipation_test_pred, dissipation_test)

    def calc_max_stress(stress):
        stress_max = np.nanmax(stress[:,:,0],axis=2)
        return stress_max
    stress_max_test = calc_max_stress(stress_test)
    stress_max_test_pred = calc_max_stress(stress_test_pred)
    stress_max_relerr = calc_relative_err(stress_max_test_pred, stress_max_test)

    # Create a new figure with two subplots
    fig, axs = plt.subplots(1, 4, figsize=(3.6*4,3.6), dpi=dpi)

    # All Stress
    ax = axs[0]
    alpha = 0.01
    color_map = ['r','r','r']
    ax.scatter(stress_test[:,0].flatten(), stress_test_pred[:,0].flatten(), c=color_map[0], alpha=alpha, s=3)
    ax.scatter(stress_test[:,1].flatten(), stress_test_pred[:,1].flatten(), c=color_map[1], alpha=alpha, s=3)
    ax.scatter(stress_test[:,2].flatten(), stress_test_pred[:,2].flatten(), c=color_map[2], alpha=alpha, s=3)
    # x, y = stress_test.flatten(), stress_test_pred.flatten()
    # not_nan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    # x, y = x[not_nan], y[not_nan]
    # h = ax.hist2d(x, y, bins=50, cmap='viridis')
    # plt.colorbar(h[3], ax=ax, label='Number of points')
    ax.set_title("All Stress (MPa)")
    ax.set_xlabel("True")
    ax.set_ylabel("Prediction")
    ax.set_xlim(-2,102)
    ax.set_ylim(-2,102)
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticks([0,25,50,75,100])
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.tick_params(direction="in",which='both')

    not_nan = np.logical_and(np.logical_not(np.isnan(stress_test.flatten())), np.logical_not(np.isnan(stress_test_pred.flatten())))
    _, _, r_value, _, _ = stats.linregress(stress_test.flatten()[not_nan], stress_test_pred.flatten()[not_nan])
    print('All Stress R2: {}'.format(r_value**2))

    what_plot = 2           # 1: scatter; 2: bar
    def group_points(x, y):
        # values of x are limited, group by x
        x_group, y_group, y_err_group = np.zeros((num_angles,)), np.zeros((num_angles,)), np.zeros((num_angles,))
        for iangle in range(num_angles):
            cur_angle = angle_unique_list[iangle]
            is_this_angle = np.array([angle == cur_angle for angle in angle_list])
            x_group[iangle] = x[is_this_angle][0]
            y_group[iangle] = y[is_this_angle].mean()
            y_err_group[iangle] = y[is_this_angle].std()
        return x_group, y_group, y_err_group

    # Energy Absorption
    ax = axs[1]
    alpha = 0.7
    color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    if what_plot == 1:
        ax.scatter(dissipation_test[:,0], dissipation_test_pred[:,0], c=color_map[0], alpha=alpha)
        ax.scatter(dissipation_test[:,1], dissipation_test_pred[:,1], c=color_map[1], alpha=alpha)
        ax.scatter(dissipation_test[:,2], dissipation_test_pred[:,2], c=color_map[2], alpha=alpha)
    else:
        x_all, y_all = [], []
        for idir in range(3):
            x, y, yerr = group_points(dissipation_test[:,idir], dissipation_test_pred[:,idir])
            ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c=color_map[idir], alpha=alpha)
            x_all.append(x)
            y_all.append(y)
        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
    ax.set_title(r"Energy Absorption (MPa)")
    ax.set_xlabel("True")
    ax.set_ylabel("Prediction")
    ax.set_xlim(-0.2,10.2)
    ax.set_ylim(-0.2,10.2)
    ax.set_xticks([0,2.5,5,7.5,10])
    ax.set_yticks([0,2.5,5,7.5,10])
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.tick_params(direction="in",which='both')
    
    _, _, r_value, _, _ = stats.linregress(dissipation_test.flatten(), dissipation_test_pred.flatten())
    print('All Energy Absorption R2: {}'.format(r_value**2))
    if what_plot == 2:
        _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
        print('Mean Energy Absorption R2: {}'.format(r_value**2))

    # Max Stress
    ax = axs[2]
    alpha = 0.7
    color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    if what_plot == 1:
        ax.scatter(stress_max_test[:,0], stress_max_test_pred[:,0], c=color_map[0], alpha=alpha)
        ax.scatter(stress_max_test[:,1], stress_max_test_pred[:,1], c=color_map[1], alpha=alpha)
        ax.scatter(stress_max_test[:,2], stress_max_test_pred[:,2], c=color_map[2], alpha=alpha)
    else:
        x_all, y_all = [], []
        for idir in range(3):
            x, y, yerr = group_points(stress_max_test[:,idir], stress_max_test_pred[:,idir])
            ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c=color_map[idir], alpha=alpha)
            x_all.append(x)
            y_all.append(y)
        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
    ax.set_title("Ultimate Stress (MPa)")
    ax.set_xlabel("True")
    ax.set_ylabel("Prediction")
    ax.set_xlim(-2,102)
    ax.set_ylim(-2,102)
    ax.set_xticks([0,25,50,75,100])
    ax.set_yticks([0,25,50,75,100])
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.tick_params(direction="in",which='both')

    _, _, r_value, _, _ = stats.linregress(stress_max_test.flatten(), stress_max_test_pred.flatten())
    print('All Max Stress R2: {}'.format(r_value**2))
    if what_plot == 2:
        _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
        print('Mean Max Stress R2: {}'.format(r_value**2))

    # Stiffness
    def linear_fit(x, y):
        x = np.tile(x, (y.shape[0], 1))
        x_mean = np.mean(x,axis=1,keepdims=True)
        y_mean = np.mean(y,axis=1,keepdims=True)
        x_bar = x - x_mean
        y_bar = y - y_mean
        slope = np.sum(y_bar * x_bar, axis=1) / np.sum(x_bar * x_bar, axis=1)
        return slope
    ax = axs[3]
    alpha = 0.7
    color_map = {0:'tab:blue',1:'tab:orange',2:'tab:green'}
    num_pts = 5
    #print(stress_test[:,0,0,1]/strain[1])
    if what_plot == 1:
        ax.scatter(linear_fit(strain[:num_pts], stress_test[:,0,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred[:,0,0,:num_pts]), c=color_map[0], alpha=alpha)
        ax.scatter(linear_fit(strain[:num_pts], stress_test[:,1,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred[:,1,0,:num_pts]), c=color_map[1], alpha=alpha)
        ax.scatter(linear_fit(strain[:num_pts], stress_test[:,2,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred[:,2,0,:num_pts]), c=color_map[2], alpha=alpha)
    else:
        x_all, y_all = [], []
        for idir in range(3):
            x0, y0 = linear_fit(strain[:num_pts], stress_test[:,idir,0,:num_pts]), linear_fit(strain[:num_pts], stress_test_pred[:,idir,0,:num_pts]),
            x, y, yerr = group_points(x0, y0)
            ax.errorbar(x, y, yerr=yerr, fmt='o', linestyle='None', capsize=2, c=color_map[idir], alpha=alpha)
            x_all.append(x)
            y_all.append(y)
        x_all = np.concatenate(x_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)
    ax.set_title("Stiffness (MPa)")
    ax.set_xlabel("True")
    ax.set_ylabel("Prediction")
    ax.set_xlim(194,506)
    ax.set_ylim(194,506)
    ax.set_xticks([200,300,400,500])
    ax.set_yticks([200,300,400,500])
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    ax.tick_params(direction="in",which='both')

    stiffness_test = np.concatenate([linear_fit(strain[:num_pts], stress_test[:,i,0,:num_pts]) for i in range(3)], axis=0)
    stiffness_test_pred = np.concatenate([linear_fit(strain[:num_pts], stress_test_pred[:,i,0,:num_pts]) for i in range(3)], axis=0)
    _, _, r_value, _, _ = stats.linregress(stiffness_test.flatten(), stiffness_test_pred.flatten())
    print('All Stiffness R2: {}'.format(r_value**2))
    if what_plot == 2:
        _, _, r_value, _, _ = stats.linregress(x_all.flatten(), y_all.flatten())
        print('Mean Stiffness R2: {}'.format(r_value**2))

    # Plot dashed diagonal lines for both subplots
    for ax in axs:
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c="black")
        ax.set_aspect('equal')

    # Adjust spacing and display the plot
    plt.tight_layout()
    plt.savefig('./paper_figure/Fig_forward_parameters.pdf')
    plt.close()


if __name__ == '__main__':
    plot_curves()
