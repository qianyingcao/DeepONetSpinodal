import os
import sys
import torch
import time
import math
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas


from mpl_toolkits import mplot3d


from config import *
from func import *
from plot import *



def read_data(return_var=True):

#     ######## Mechanics ########

    df = pandas.read_excel('../../../inverse_stress_target.xlsx', sheet_name='Sheet1', engine='openpyxl')
    data_array = df.to_numpy()
    strain = data_array[:,0]
    stress_out_all= data_array[:,1:].transpose()

    stress_out_targets = np.zeros((6, 3, 2, 51))
    for angle in range(6):
        for direction in range(3):
            for stage in range(2):
                original_row_index = angle*6 + direction*2 + stage
                stress_out_targets[angle, direction, stage] = stress_out_all[original_row_index,:]

    ######## Design Candidate Pool ########

    folders = ['../../../23C_Data_3D_50_invpool_grid7d5_perm_type1/',
               '../../../23C_Data_3D_50_invpool_grid7d5_perm_type2/',
               '../../../23C_Data_3D_50_invpool_grid7d5_perm_type3/' ]
    files = [os.listdir(folder) for folder in folders]
    is_first = True
    phase_pool_all = []
    angle_pool_all = []
    sample_id_pool_all = []
    for idx_folder, folder in enumerate(folders):
        for idx_file, file in enumerate(files[idx_folder]):
            if file.startswith('Spinodal'):
                print(file)
                splitted = file.split('_')
                angles_pool = tuple([int(math.ceil(float(item))) for item in splitted[2:5]])
                sample_id_pool = int(splitted[-1][:-4])
                #print('Angle: {}, Sample ID: {}'.format(angles, sample_id))
                data_geom = io.loadmat(folder+file)
                phase_pool = data_geom['is_solid_3d'].transpose((1,0,2))     # matlab meshgrid 3d yields (y, x, z)

                phase_pool_all.append(phase_pool)
                angle_pool_all.append(angles_pool)
                sample_id_pool_all.append(sample_id_pool)

    npize = lambda x: [np.array(item) for item in x]
    stress_out_targets, phase_pool_all, angle_pool_all, sample_id_pool_all = \
            npize([stress_out_targets, phase_pool_all, angle_pool_all, sample_id_pool_all])
    
    num_imag_pool_samples = phase_pool_all.shape[0]
    perm_pool = np.random.permutation(num_imag_pool_samples)
    permize_pool = lambda x: [item[perm_pool] for item in x]
    phase_pool_all, angle_pool_all, sample_id_pool_all = permize_pool([phase_pool_all, angle_pool_all, sample_id_pool_all])
    
    print('========== Summary of Candidate Pool ==========')
    print('Phase Data Shape: {}'.format(phase_pool_all.shape))
    print('Angle Data Shape: {}'.format(angle_pool_all.shape))
    print('Sample ID Data Shape: {}'.format(sample_id_pool_all.shape))

    #print('========== Summary of Used Data (Mechanics & Phase) ==========')

    # for item in [stress_all, stress_weight_all, phase_all, angle_all]:
    #     if np.isnan(item).any():
    #         raise ValueError('There exists NaN in a data array. Check and Debug it.')

    if return_var: return strain, stress_out_targets, phase_pool_all, angle_pool_all, sample_id_pool_all

if __name__ == '__main__':
    read_data(return_var=False)
