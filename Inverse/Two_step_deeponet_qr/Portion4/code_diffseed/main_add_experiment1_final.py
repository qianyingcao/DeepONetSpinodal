import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import time
import math
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import sys
from scipy import io
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas

from func import *
from config import *
#from plot import *
import pickle
from NN_QR import predict_qr

os.makedirs('./{}/Fig/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Model/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Output/'.format(CASE_NAME),exist_ok=True)


df = pandas.read_excel('experiment1.xlsx', sheet_name='1', engine='openpyxl')
data_array = df.to_numpy()
experiment1x = data_array[:,0:2]
experiment1y = data_array[:,2:4]
experiment1z = data_array[:,4:6]
experiment2x = data_array[:,6:8]
experiment2y = data_array[:,8:10]
experiment2z = data_array[:,10:12]
experiment3x = data_array[:,12:14]
experiment3y = data_array[:,14:16]
experiment3z = data_array[:,16:18]
experiment4x = data_array[:,18:20]
experiment4y = data_array[:,20:22]
experiment4z = data_array[:,22:24]
experiment5x = data_array[:,24:26]
experiment5y = data_array[:,26:28]
experiment5z = data_array[:,28:30]
experiment6x = data_array[:,30:32]
experiment6y = data_array[:,32:34]
experiment6z = data_array[:,34:36]
experiment7x = data_array[:,36:38]
experiment7y = data_array[:,38:40]
experiment7z = data_array[:,40:42]

#data = read_data()
data = torch.load('data')
strain, stress_out_targets0, phase_pool_all, angle_pool_all, sample_id_pool_all = data
stress_out_targets = np.concatenate((stress_out_targets0[:3,:,:,:],stress_out_targets0[4:,:,:,:]), axis=0)

print('Target stress shape:{}'.format(stress_out_targets.shape))
#torch.save(data,'data')
print('Strain Shape: {}'.format(strain.shape))
print('Stress Out Targets Shape: {}'.format(stress_out_targets.shape))
print('Phase Pool All Shape: {}'.format(phase_pool_all.shape))
print('Angle Pool All Shape: {}'.format(angle_pool_all.shape))
print('Sample ID Pool All Shape: {}'.format(sample_id_pool_all.shape))
######### IMPORTANT ###########
model_path_list =  [
                    ("./model_qr_r1/model.0", "./model_qr_b1/model.0", "./model_qr_b1/Rmatrix1", "./model_qr_b1/Rmatrix2", "./model_qr_b1/Rmatrix3"),
                    ("./model_qr_r2/model.0", "./model_qr_b2/model.0", "./model_qr_b2/Rmatrix1", "./model_qr_b2/Rmatrix2", "./model_qr_b2/Rmatrix3"),
                    ("./model_qr_r3/model.0", "./model_qr_b3/model.0", "./model_qr_b3/Rmatrix1", "./model_qr_b3/Rmatrix2", "./model_qr_b3/Rmatrix3"),
                    ("./model_qr_r4/model.0", "./model_qr_b4/model.0", "./model_qr_b4/Rmatrix1", "./model_qr_b4/Rmatrix2", "./model_qr_b4/Rmatrix3"),
                    ("./model_qr_r5/model.0", "./model_qr_b5/model.0", "./model_qr_b5/Rmatrix1", "./model_qr_b5/Rmatrix2", "./model_qr_b5/Rmatrix3"),
                    ("./model_qr_r6/model.0", "./model_qr_b6/model.0", "./model_qr_b6/Rmatrix1", "./model_qr_b6/Rmatrix2", "./model_qr_b6/Rmatrix3")
                    ]

def forward_QR_DeepONet(phase_2d_pt, model_path):
    
    filename_t, filename_b, filename_R1, filename_R2, filename_R3 = model_path

    paramsb = load_model(filename_b)
    paramst = load_model(filename_t)
    R1 = load_model(filename_R1)
    R2 = load_model(filename_R2)
    R3 = load_model(filename_R3)
    R1inv=torch.linalg.inv(R1)
    R2inv=torch.linalg.inv(R2)
    R3inv=torch.linalg.inv(R3)

    stages = np.linspace(-1,1,NUM_STAGES)
    strain_all, stages_all = np.meshgrid(strain,stages,indexing='xy')
    strain_all, stages_all = strain_all.reshape([-1,1]), stages_all.reshape([-1,1])
    strain_all, stages_all = totorch(strain_all), totorch(stages_all)
    strain_2 = torch.concat((strain_all, stages_all),dim=1)

    stress_pred = predict_qr(paramsb, paramst, R1inv, R2inv, R3inv, [phase_2d_pt, strain_2])* totorch(strain[None,None,None,:])*NORMALIZER
    return tonumpy(stress_pred)


def load_model(filename):
    with open(filename, 'rb') as file:
        param=pickle.load(file)
    return param

model_fwd_functions =  [forward_QR_DeepONet,
                        forward_QR_DeepONet,
                        forward_QR_DeepONet,
                        forward_QR_DeepONet,
                        forward_QR_DeepONet,
                        forward_QR_DeepONet
]


assert len(model_path_list) == len(model_fwd_functions)

num_models = len(model_path_list)
#################################
num_cs = 7      # according to how the forward model is trained; choose from the full 3D geometry to these slices
image_size = phase_pool_all.shape[1]      # according to how the forward model is trained
cs_ids = np.zeros(num_cs,dtype=int)
cs_xs = np.zeros(num_cs)
for ics in range(num_cs):
    cs_ids[ics] = int(np.floor((image_size-1)*(ics+1)/(num_cs+1)))
    cs_xs[ics] = (cs_ids[ics])/(image_size-1)*1

xs = np.linspace(0.0, 1.0, image_size)
X_z, Y_z, Z_z = np.meshgrid(xs,xs,cs_xs)        # xy, not ij -- hence the three axes are (y,x,z)
X_y, Z_y, Y_y = np.meshgrid(xs,xs,cs_xs)
Y_x, Z_x, X_x = np.meshgrid(xs,xs,cs_xs)

phase_pool_z = phase_pool_all[:,:,:,cs_ids].transpose((0, 2, 1, 3))       # (batch, x, y, z) --> (batch, y, x, z)
phase_pool_y = phase_pool_all[:,:,cs_ids,:].transpose((0, 3, 1, 2))       # (batch, x, y, z) --> (batch, z, x, y)
phase_pool_x = phase_pool_all[:,cs_ids,:,:].transpose((0, 3, 2, 1))       # (batch, x, y, z) --> (batch, z, y, x)
phase_pool_2d_slices = np.concatenate((phase_pool_z[:,:,:,:,None], phase_pool_y[:,:,:,:,None], phase_pool_x[:,:,:,:,None]), axis=4)
phase_pool_2d_slices = phase_pool_2d_slices[...,[2, 1, 0]]
print('Phase Pool 2D Slices Shape: {}'.format(phase_pool_2d_slices.shape))
num_samples = phase_pool_2d_slices.shape[0]
batches, num_batches = batch_info(num_samples)

stress_pool_pred = []
stress_pool_pred_std = []

for ibatch in range(num_batches):
    istart = batches[ibatch,0]
    iend = batches[ibatch,1]
    phase_pool_2d_slices_batch = phase_pool_2d_slices[istart:iend]
    phase_pool_2d_slices_batch_pt = totorch(phase_pool_2d_slices_batch.reshape([phase_pool_2d_slices_batch.shape[0],
                                                                                phase_pool_2d_slices_batch.shape[1],
                                                                                phase_pool_2d_slices_batch.shape[2], -1]).transpose((0,3,1,2)))
    ###### IMPORTANT ######
    stress_pool_batch_pred_all_models = []
    for idx_model in range(num_models):
        model_path = model_path_list[idx_model]
        model_fwd_function = model_fwd_functions[idx_model]
        stress_pool_batch_pred_model_i = model_fwd_function(phase_pool_2d_slices_batch_pt, model_path)
        stress_pool_batch_pred_all_models.append(stress_pool_batch_pred_model_i)
    stress_pool_batch_pred = np.stack(stress_pool_batch_pred_all_models, axis=0).mean(axis=0)       # Or some other more complicated functions to better handle (or remove) extreme cases
    stress_pool_batch_std = np.std(np.stack(stress_pool_batch_pred_all_models, axis=0), axis=0)
    #print(stress_pool_batch_std)
    stress_pool_pred.append(stress_pool_batch_pred)
    stress_pool_pred_std.append(stress_pool_batch_std)
    ########################

stress_pool_pred = np.concatenate(stress_pool_pred, axis=0)
stress_pool_pred_std = np.concatenate(stress_pool_pred_std, axis=0)
# print(stress_pool_pred.shape)
#print(stress_pool_pred_std)

num_designs = 7
idx_best_design_all = []
for idesign in range(num_designs):
    stress_out_target = stress_out_targets[idesign]
    design_errors = np.sqrt(np.nanmean((stress_pool_pred - stress_out_target[None])**2, axis=(1,2,3)))
    idx_best_design = np.argmin(design_errors)
    print('Min L2 err: {}'.format(np.min(design_errors)))
    idx_best_design_all.append(idx_best_design)
idx_best_design_all = np.array(idx_best_design_all)
angle_underlying_best_design_all = angle_pool_all[idx_best_design_all]
stress_best_design_all = stress_pool_pred[idx_best_design_all]
stress_best_design_std_all = stress_pool_pred_std[idx_best_design_all]
nan_indices = np.isnan(stress_out_targets)
stress_best_design_all[nan_indices] = np.nan
stress_best_design_std_all[nan_indices] = np.nan
phase_best_design_all = phase_pool_all[idx_best_design_all]
#print(stress_out_targets)

io.savemat('data_final_design.mat', 
            mdict={'strain':strain,
                    'stress_out_targets':stress_out_targets,
                    'stress_best_design_all':stress_best_design_all,
                    'stress_best_design_std_all':stress_best_design_std_all})

print('=====Compare Designed & Actual Underlying Angles=====')
print(angle_underlying_best_design_all)


### Stress-Strain Curve Comparison of Design ###
fig, axs = plt.subplots(nrows = num_designs, ncols = 3, figsize = (3.2*3,3.0*num_designs))
L2_err_all = []
L2_relerr_all = []
for iplot in range(num_designs):
    def single_subplot(exp,y, y_pred, y_std, irow, icol):
        color_map = {0:'b',1:'r',2:'g'}
        strain_plot = strain
        axs[irow,icol].plot(strain_plot, y[0], '-'+color_map[icol])
        axs[irow,icol].plot(strain_plot, y[1], '-'+color_map[icol])
        axs[irow,icol].plot(strain_plot, y_pred[0], '--'+color_map[icol])
        axs[irow,icol].plot(strain_plot, y_pred[1], '--'+color_map[icol])
        axs[irow,icol].fill_between(strain_plot, y_pred[0] + 2*y_std[0], y_pred[0] - 2*y_std[0], alpha=0.3, color=color_map[icol])
        axs[irow,icol].fill_between(strain_plot, y_pred[1] + 2*y_std[1], y_pred[1] - 2*y_std[1], alpha=0.3, color=color_map[icol])
        axs[irow,icol].plot(exp[:,0], exp[:,1], '-.k', linewidth=2)
        axs[irow,icol].set_xlabel('Strain')
        if icol == 0:
            axs[irow,icol].set_ylabel('Stress (MPa)')
        if irow == 0 & icol == 0:
            axs[irow,icol].legend(['Target', 'Pred', 'Exp'])

        return
    stress_target_plot = stress_out_targets[iplot]
    stress_pred_plot = stress_best_design_all[iplot]
    stress_pred_std_plot = stress_best_design_std_all[iplot]
    if iplot == 0:
        single_subplot(experiment1x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment1y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment1z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 1:
        single_subplot(experiment2x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment2y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment2z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 2:
        single_subplot(experiment3x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment3y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment3z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 3:
        single_subplot(experiment4x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment4y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment4z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 4:
        single_subplot(experiment5x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment5y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment5z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 5:
        single_subplot(experiment6x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment6y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment6z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    elif iplot == 6:
        single_subplot(experiment7x, stress_target_plot[0], stress_pred_plot[0], stress_pred_std_plot[0], iplot, 0)
        single_subplot(experiment7y, stress_target_plot[1], stress_pred_plot[1], stress_pred_std_plot[1], iplot, 1)
        single_subplot(experiment7z, stress_target_plot[2], stress_pred_plot[2], stress_pred_std_plot[2], iplot, 2)
    
    diff = stress_pred_plot - stress_target_plot
    print(stress_pred_plot.shape)
    print(stress_target_plot.shape)
    L2_err = np.sqrt(np.nanmean(diff**2))
    L2_relerr = np.sqrt(np.nanmean(diff**2)) / np.sqrt(np.nanmean(stress_target_plot**2))
    L2_err_all.append(L2_err)
    L2_relerr_all.append(L2_relerr)

    saved_mat = {'strain':strain, 'stress_target':stress_target_plot, 'stress_pred':stress_pred_plot}
    io.savemat('./{}/Output/curve_{}.mat'.format(CASE_NAME, iplot+1), saved_mat)
#plt.show()

L2_err_mean = np.mean(L2_err_all)
L2_relerr_mean = np.mean(L2_relerr_all)

print('L2 Rel. Err.')
print(L2_relerr_all)


fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./Design.png')
plt.close()


for idesign in range(num_designs):
    geo = phase_best_design_all[idesign]
    stress = stress_best_design_all[idesign]
    std_stress = stress_best_design_std_all[idesign]
    angle_picked = angle_underlying_best_design_all[idesign]
    io.savemat('./{}/Output/geometry_{}.mat'.format(CASE_NAME, idesign+1), {'phase': geo})
    io.savemat('./{}/Output/info_{}.mat'.format(CASE_NAME, idesign+1), {'stress':stress, 'angle_picked':angle_picked,'std':std_stress})