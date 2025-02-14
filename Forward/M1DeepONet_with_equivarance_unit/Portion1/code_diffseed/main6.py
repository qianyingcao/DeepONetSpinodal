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
import scipy

from NN import *
from func import *
from config import *
from plot import *

seed = 7891
np.random.seed(seed)
torch.manual_seed(seed)

Case_name = 'onlycase6'
os.makedirs('./{}/Fig/'.format(Case_name),exist_ok=True)
os.makedirs('./{}/Model/'.format(Case_name),exist_ok=True)
os.makedirs('./{}/Output/'.format(Case_name),exist_ok=True)

# Import Data

data = io.loadmat('./data6.mat')
strain = data['strain']
stress_train = data['stress_train']
stress_test = data['stress_test']
stress_weight_train = data['stress_weight_train']
stress_weight_test = data['stress_weight_test']
phase_train = data['phase_train']
phase_test = data['phase_test']
angle_train = data['angle_train']
angle_test = data['angle_test']
num_pts = data['num_pts']
num_cs = data['num_cs']
strain = strain.squeeze()
num_cs = num_cs[0,0]
num_pts = num_pts[0,0]


## initialization
model = NN(strain, num_cs)
model.to(DEVICE)

loss_train_hist = []
loss_test_hist = []
saved = {}

L2_err_hist = []
L2_relerr_hist = []

start_time = time.time()
# Training
for iepoch in range(-1,NUM_EPOCHS):
    
    loss_train, stress_train_pred, dt, t1, t2, t3 = model.iterate_once(iepoch, phase_train, stress_train, stress_weight_train)
    epoch_time = time.time() - start_time
    loss_train_hist.append(loss_train)
    

    loss_test, stress_test_pred, t1, t2, t3 = model.test(phase_test, stress_test, stress_weight_test)
    loss_test_hist.append(loss_test)
    
    # Print #
    print('Epoch: {}; T: {:0.6f}; Loss train: {:.4f} test: {:.4f}'.format(iepoch+1, epoch_time, loss_train, loss_test), flush=True)

    # ### Plot ###
    # if (iepoch+1 in PLOT_ITS):

    #     ### Stress-Strain Curve Comparison ###
    #     fig, axs = plt.subplots(nrows = NUM_SUBPLOT, ncols = 3, figsize = (3.2*3,3.0*NUM_SUBPLOT))
    #     L2_err_all = []
    #     L2_relerr_all = []
    #     for iplot in range(NUM_SUBPLOT):
    #         def single_subplot(y, y_pred, irow, icol, title):
    #             color_map = {0:'b',1:'r',2:'g'}
    #             for istage in range(NUM_STAGES):
    #                 strain_plot = strain
    #                 axs[irow,icol].plot(strain_plot, y_pred[istage,:], '--'+color_map[icol])
    #                 axs[irow,icol].plot(strain_plot, y[istage,:], '-'+color_map[icol])
    #             axs[irow,icol].set_title(title)
    #             return
    #         stress_test_plot = np.where(stress_test[iplot]==NEGATIVE_NUMBER, np.nan, stress_test[iplot])
    #         def plot_process(stress):
    #             # stress: (3, num_stages, num_pts)
    #             stress_copy = np.copy(stress)
    #             for idir in range(3):
    #                 for istage in range(NUM_STAGES):
    #                     if istage%2 == 1:
    #                         stress_i = stress_copy[idir,istage]
    #                         pointer = stress_i.shape[0]-1
    #                         while pointer>=0 and (not stress_i[pointer]<0):
    #                             pointer -= 1
    #                         pointer += 1
    #                         stress_copy[idir,istage,:pointer] = 0
    #             return stress_copy
    #         stress_test_pred_plot = plot_process(stress_test_pred[iplot])
    #         single_subplot(stress_test_plot[0], stress_test_pred_plot[0], iplot, 0, 'angle {} x'.format(angle_test[iplot]))
    #         single_subplot(stress_test_plot[1], stress_test_pred_plot[1], iplot, 1, 'angle {} y'.format(angle_test[iplot]))
    #         single_subplot(stress_test_plot[2], stress_test_pred_plot[2], iplot, 2, 'angle {} z'.format(angle_test[iplot]))
    #         diff = stress_test_pred[iplot] - stress_test[iplot]
    #         L2_err = np.sqrt(np.nanmean(diff**2))
    #         L2_relerr = np.sqrt(np.nanmean(diff**2)) / np.sqrt(np.nanmean(stress_test[iplot]**2))
    #         #single_subplot(diff, iplot, 2, 'L2 Err: {:.4f}; L2 Rel Err: {:.4f}'.format(L2_err, L2_relerr))
    #         L2_err_all.append(L2_err)
    #         L2_relerr_all.append(L2_relerr)

    #     L2_err_mean = np.mean(L2_err_all)
    #     L2_relerr_mean = np.mean(L2_relerr_all)

    #     L2_err_hist.append(L2_err_mean)
    #     L2_relerr_hist.append(L2_relerr_mean)


    #     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     # fig.suptitle('Training Epoch #{} (L2 Err: {:.4f}; Rel Err: {:.4f})'.format(iepoch+1, L2_err, L2_relerr))
    #     # plt.savefig('./{}/Fig/Fig_Curve_Comparison_epoch{}.png'.format(CASE_NAME, iepoch+1))
    #     # plt.show()

    #     ###### Loss ######
    #     plt.figure(2)
    #     plt.semilogy(np.arange(iepoch+2),np.array(loss_train_hist),'-b', label='Train')
    #     plt.semilogy(np.arange(iepoch+2),np.array(loss_test_hist),'-r', label='Test')
    #     plt.legend()
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.tight_layout()
    #     plt.savefig('./{}/Fig/Fig_loss_epoch{}.png'.format(Case_name, iepoch+1))
    #     plt.show()

    if (iepoch+1 in SAVE_ITS):

        np.save('./{}/Output/SavedOutputs_{}.npy'.format(Case_name, iepoch+1),
                {'stress_train_pred':stress_train_pred,
                 'stress_test_pred':stress_test_pred})
        # torch.save(model, './{}/Model/Model_epoch{}.pt'.format(CASE_NAME, iepoch+1))
    start_time = time.time()

saved = {}

saved['data'] = (strain, stress_test, stress_test_pred, stress_weight_test, angle_test)
saved['loss_train'] = np.array(loss_train_hist)
saved['loss_test'] = np.array(loss_test_hist)
saved['L2_err'] = np.array(L2_err_hist)
saved['L2_relerr'] = np.array(L2_relerr_hist)
saved['SAVE_ITS'] = SAVE_ITS
saved['PLOT_ITS'] = PLOT_ITS

np.save('./{}/Output/SavedOutputs_final.npy'.format(Case_name), saved)

# ====================================
# saving settings
# ====================================
save_index=6
current_directory = os.getcwd()
case = "Case_"
folder_index = str(save_index)

results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)

    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)


torch.save(model, save_models_to+'Metamaterial')
scipy.io.savemat(save_results_to+'pred_test_data.mat', 
                    mdict={'stress_test': stress_test, 
                            'stress_pred': stress_test_pred})  


U1,S1,V1 = torch.linalg.svd(t1,full_matrices=False)
U2,S2,V2 = torch.linalg.svd(t2,full_matrices=False)
U3,S3,V3 = torch.linalg.svd(t3,full_matrices=False)
################################################
phi = torch.concat((t1, t2, t3),dim=1)
U,S,V = torch.linalg.svd(phi,full_matrices=False)
save_results_to = './basis/'
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)
io.savemat(save_results_to+'Basis_original_method.mat',
            mdict={ 'U1': U1.detach().cpu().numpy(),
                    'U2': U2.detach().cpu().numpy(),
                    'U3': U3.detach().cpu().numpy(),
                    'U': U.detach().cpu().numpy()})


plt.figure(1)
plt.semilogy(np.arange(iepoch+2),np.array(loss_train_hist),'-b', label='Train')
plt.semilogy(np.arange(iepoch+2),np.array(loss_test_hist),'-r', label='Test')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('./{}/Fig/Fig_loss_epoch{}.png'.format(Case_name, iepoch+1))
plt.show()
