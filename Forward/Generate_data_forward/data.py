import os
import sys
import torch
import time
import math
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d


from config import *
from func import *
from plot import *


def expand_data(X_raw, y_raw, y_weight_raw, param_raw):
    # X: (90, 101, 101, 3/7, 3)
    # y: (90, 3, 2, 31)

    print('expanding data...')

    perms = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
    num_perms = len(perms)
    X = np.zeros((0,X_raw.shape[1],X_raw.shape[2],X_raw.shape[3],X_raw.shape[4]))
    y = np.zeros((0,y_raw.shape[1],y_raw.shape[2],y_raw.shape[3]))
    y_weight = np.zeros((0, y_weight_raw.shape[1],y_weight_raw.shape[2],y_weight_raw.shape[3]))
    param = np.zeros((0, param_raw.shape[1]))

    for iperm_cs in range(num_perms):
        perm_i = perms[iperm_cs]
        X = np.concatenate((X,np.copy(X_raw)[:,:,:,:,perm_i]),axis=0)
        y = np.concatenate((y,np.copy(y_raw)[:,perm_i,:,:]),axis=0)
        y_weight = np.concatenate((y_weight,np.copy(y_weight_raw)[:,perm_i,:,:]),axis=0)
        param = np.concatenate((param, np.copy(param_raw)[:,perm_i]), axis=0)

    print('expanding data completed')

    return X, y, y_weight, param

#     ######## Mechanics ########

data_mechanics = np.load('data_mechanics.npy', allow_pickle=True)
stress_raw = data_mechanics.item()['stress']
strain = data_mechanics.item()['strain']
param = data_mechanics.item()['angle']
directions = data_mechanics.item()['direction']

num_mech_samples0 = stress_raw.shape[0]
stress_group_x = []
stress_group_y = []
stress_group_z = []
stress_group1_x = []
stress_group1_y = []
stress_group1_z = []
stress_group2_x = []
stress_group2_y = []
stress_group2_z = []
stress_group3_x = []
stress_group3_y = []
stress_group3_z = []
stress_group4_x = []
stress_group4_y = []
stress_group4_z = []
stress_group5_x = []
stress_group5_y = []
stress_group5_z = []
stress_group6_x = []
stress_group6_y = []
stress_group6_z = []
stress_group7_x = []
stress_group7_y = []
stress_group7_z = []
param_group = []
param_group_uncertain = ANGLES_uncertain
for isample in range(num_mech_samples0):
    angle_tuple =[int(item) for item in param[isample]]
    if angle_tuple not in ANGLES_uncertain:
        if angle_tuple not in param_group:
            param_group.append(angle_tuple)
        if directions[isample] == 'x':
            stress_group_x.append(stress_raw[isample])
        elif directions[isample] == 'y':
            stress_group_y.append(stress_raw[isample])
        elif directions[isample] == 'z':
            stress_group_z.append(stress_raw[isample])
    else:
        if angle_tuple == [0,0,20]:
            if directions[isample] == 'x':
                stress_group1_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group1_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group1_z.append(stress_raw[isample])
        elif angle_tuple == [0, 0, 30]:
            if directions[isample] == 'x':
                stress_group2_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group2_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group2_z.append(stress_raw[isample])
        elif angle_tuple == [0, 30, 30]:
            if directions[isample] == 'x':
                stress_group3_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group3_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group3_z.append(stress_raw[isample])
        elif angle_tuple == [30, 30, 30]:
            if directions[isample] == 'x':
                stress_group4_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group4_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group4_z.append(stress_raw[isample])
        elif angle_tuple == [0, 30, 60]:
            if directions[isample] == 'x':
                stress_group5_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group5_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group5_z.append(stress_raw[isample])
        elif angle_tuple == [0, 45, 60]:
            if directions[isample] == 'x':
                stress_group6_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group6_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group6_z.append(stress_raw[isample])
        elif angle_tuple == [15, 15, 30]:
            if directions[isample] == 'x':
                stress_group7_x.append(stress_raw[isample])
            elif directions[isample] == 'y':
                stress_group7_y.append(stress_raw[isample])
            elif directions[isample] == 'z':
                stress_group7_z.append(stress_raw[isample])


stress_group_x = np.array(stress_group_x).reshape(-1,1,2,stress_raw.shape[2])
stress_group_y = np.array(stress_group_y).reshape(-1,1,2,stress_raw.shape[2])
stress_group_z = np.array(stress_group_z).reshape(-1,1,2,stress_raw.shape[2])
stress_group = np.concatenate([stress_group_x, stress_group_y, stress_group_z], axis=1)
param_new = np.concatenate([param_group, param_group_uncertain], axis=0)

index = [[len(stress_group1_x), len(stress_group1_y), len(stress_group1_z)],
            [len(stress_group2_x), len(stress_group2_y), len(stress_group2_z)],
            [len(stress_group3_x), len(stress_group3_y), len(stress_group3_z)],
            [len(stress_group4_x), len(stress_group4_y), len(stress_group4_z)],
            [len(stress_group5_x), len(stress_group5_y), len(stress_group5_z)],
            [len(stress_group6_x), len(stress_group6_y), len(stress_group6_z)],
            [len(stress_group7_x), len(stress_group7_y), len(stress_group7_z)]]


for ii in range(6):
    #random_index = np.random.randint(0,index)
    #np.save('random_index%d.npy'% (ii+1), random_index)
    random_index = np.load('random_index%d.npy'% (ii+1))
    stress_group1_x_sam=stress_group1_x[random_index[0,0]]
    stress_group1_y_sam=stress_group1_y[random_index[0,1]]
    stress_group1_z_sam=stress_group1_z[random_index[0,2]]
    stress_group2_x_sam=stress_group2_x[random_index[1,0]]
    stress_group2_y_sam=stress_group2_y[random_index[1,1]]
    stress_group2_z_sam=stress_group2_z[random_index[1,2]]
    stress_group3_x_sam=stress_group3_x[random_index[2,0]]
    stress_group3_y_sam=stress_group3_y[random_index[2,1]]
    stress_group3_z_sam=stress_group3_z[random_index[2,2]]
    stress_group4_x_sam=stress_group4_x[random_index[3,0]]
    stress_group4_y_sam=stress_group4_y[random_index[3,1]]
    stress_group4_z_sam=stress_group4_z[random_index[3,2]]
    stress_group5_x_sam=stress_group5_x[random_index[4,0]]
    stress_group5_y_sam=stress_group5_y[random_index[4,1]]
    stress_group5_z_sam=stress_group5_z[random_index[4,2]]
    stress_group6_x_sam=stress_group6_x[random_index[5,0]]
    stress_group6_y_sam=stress_group6_y[random_index[5,1]]
    stress_group6_z_sam=stress_group6_z[random_index[5,2]]
    stress_group7_x_sam=stress_group7_x[random_index[6,0]]
    stress_group7_y_sam=stress_group7_y[random_index[6,1]]
    stress_group7_z_sam=stress_group7_z[random_index[6,2]]
    
    stress_group1 = np.concatenate([stress_group1_x_sam[None,:], stress_group1_y_sam[None,:], stress_group1_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group2 = np.concatenate([stress_group2_x_sam[None,:], stress_group2_y_sam[None,:], stress_group2_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group3 = np.concatenate([stress_group3_x_sam[None,:], stress_group3_y_sam[None,:], stress_group3_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group4 = np.concatenate([stress_group4_x_sam[None,:], stress_group4_y_sam[None,:], stress_group4_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group5 = np.concatenate([stress_group5_x_sam[None,:], stress_group5_y_sam[None,:], stress_group5_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group6 = np.concatenate([stress_group6_x_sam[None,:], stress_group6_y_sam[None,:], stress_group6_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group7 = np.concatenate([stress_group7_x_sam[None,:], stress_group7_y_sam[None,:], stress_group7_z_sam[None,:]], axis=0).reshape(-1,3,2,stress_raw.shape[2])
    stress_group_uncertan = np.concatenate([stress_group1, stress_group2, stress_group3, stress_group4, stress_group5, stress_group6, stress_group7], axis=0)
    stress_sample = np.concatenate([stress_group, stress_group_uncertan], axis=0)

    # Norm & Average
    num_mech_samples = stress_sample.shape[0]
    #stress = np.zeros_like(stress_sample)
    #for isample in range(num_mech_samples):
     #   angle_tuple = tuple([int(item) for item in param_new[isample]])
        #stress[isample] = average_xyz(angle_tuple, stress_sample[isample])
    stress = stress_sample
    stress = stress / 50
    # Pad NaN
    stress[:,:,:,0] = np.where(np.isnan(stress[:,:,:,0]), 0.0, stress[:,:,:,0])
    stress_weight = np.where(np.isnan(stress), 0.0, 1.0)
    stress = np.where(np.isnan(stress), NEGATIVE_NUMBER, stress)
    # Save
    data_mechanics_dict = {}
    for isample in range(num_mech_samples):
        angle_tuple = tuple([int(item) for item in param_new[isample]])
        data_mechanics_dict[angle_tuple] = [stress[isample], stress_weight[isample]]

    print('========== Summary of Mechanics Data (Pre) ==========')
    print('Param: (shape: {})'.format(np.array(param).shape))
    param = np.array(param_new, dtype=int)
    #param = [tuple([int(item2) for item2 in item]) for item in param]
    print(param_new)

    print('Strain: (shape: {})'.format(strain.shape))
    print(strain)
    print('Stress shape: {}'.format(stress.shape))

    ######## Geometry & Actual Data ########


    path = 'Data_CrossSec_2D_50_newData/'
    files = os.listdir(path)
    is_first = True
    phase_all = []
    stress_all = []
    stress_weight_all = []
    angle_all = []
    sample_id_all = []
    for file in files:
        if file.startswith('Spinodal'):
            splitted = file.split('_')
            angles = tuple([int(item) for item in splitted[2:5]])
            sample_id = int(splitted[-1][:-4])
            if angles in param_new:
                data_geom = io.loadmat(path+file)
                
                phase = data_geom['is_solid'][:,:,:,:][...,[2, 1, 0]]   # Choose all seven cross sections in each direction
                #print(phase.shape)

                if is_first:
                    num_pts = phase.shape[0]
                    num_cs = phase.shape[2]

                phase_all.append(phase)

                stress_all.append(data_mechanics_dict[angles][0])
                stress_weight_all.append(data_mechanics_dict[angles][1])
                angle_all.append(angles)
                sample_id_all.append(sample_id)

            else:
                print('{} is not in ANGLES_ALL, not read in!'.format(angles))
                

    phase_all = np.array(phase_all)
    stress_all = np.array(stress_all)
    stress_weight_all = np.array(stress_weight_all)
    sample_id_all = np.array(sample_id_all)

    print('========== Summary of Actual Data (Mechanics & Phase) ==========')
    print('Phase Data Shape: {}'.format(phase_all.shape))
    print('Mechanics Data Shape: {}'.format(stress_all.shape))
    print('Mechanics Weight Data Shape: {}'.format(stress_weight_all.shape))
    print('Angle Data Shape: {}'.format(np.array(angle_all).shape))


    angle_all = np.array(angle_all)

    #phase_all, stress_all, stress_weight_all, angle_all = expand_data(phase_all, stress_all, stress_weight_all, angle_all)
    num_imag_samples = phase_all.shape[0]
    print('========== Summary of Actual Data (Expanded) ==========' )
    print('Phase Data Shape: {}'.format(phase_all.shape))
    print('Mechanics Data Shape: {}'.format(stress_all.shape))
    print('Mechanics Weight Data Shape: {}'.format(stress_weight_all.shape))
    print('Angle Data Shape: {}'.format(angle_all.shape))

    num_test = int(PCT_TEST*num_imag_samples)
    num_train = num_imag_samples - num_test
    # perm0 = np.random.permutation(num_imag_samples)
    # perm = np.random.permutation(num_imag_samples)
    # np.save('perm.npy', perm)
    # np.save('perm0.npy', perm0)
    perm = np.load('perm.npy')
    perm0 = np.load('perm0.npy')

    is_test = (np.arange(num_imag_samples) < num_test)[perm0]
    is_train = np.logical_not(is_test)

    phase_all = phase_all[perm]
    stress_all = stress_all[perm]
    stress_weight_all = stress_weight_all[perm]
    angle_all = angle_all[perm]

    stress_train = stress_all[is_train][::PORTION]
    stress_test = stress_all[is_test][::1]
    print('Stress train & test shapes: {}, {}'.format(stress_train.shape, stress_test.shape))

    stress_weight_train = stress_weight_all[is_train][::PORTION]
    stress_weight_test = stress_weight_all[is_test][::1]
    print('Stress Weight train & test shapes: {}, {}'.format(stress_weight_train.shape, stress_weight_test.shape))

    phase_train = phase_all[is_train][::PORTION,::DOWN_SAMPLE,::DOWN_SAMPLE]
    phase_test = phase_all[is_test][::1,::DOWN_SAMPLE,::DOWN_SAMPLE]
    print('Phase train & test shapes: {}, {}'.format(phase_train.shape, phase_test.shape))

    angle_train = angle_all[is_train][::PORTION]
    angle_test = angle_all[is_test][::1]
    print('Angle train & test shapes: {}, {}'.format(angle_train.shape, angle_test.shape))
    print('Strain Shape: {}'.format(strain.shape))
    print(num_cs)
    print(num_pts)

    io.savemat('data%d.mat'% (ii+1), 
               mdict={'strain':strain, 
                      'stress_train':stress_train, 
                      'stress_test':stress_test, 
                      'stress_weight_train':stress_weight_train, 
                      'stress_weight_test':stress_weight_test, 
                      'phase_train':phase_train, 
                      'phase_test':phase_test, 
                      'angle_train':angle_train, 
                      'angle_test':angle_test, 
                      'num_pts':num_pts, 
                      'num_cs':num_cs,
                      'random_index':random_index})
    

