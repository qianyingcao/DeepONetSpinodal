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
import pickle

from func import *
from config import *
#from plot import *


# Import trained forward model
##################################
# Import 3D Data (previously generated for inverse)


def conv_layer(x, filter_size, stride, padding, w, b):
        layer = nn.functional.conv2d(x, w, stride=stride, padding=padding)
        layer += b.view(1, -1, 1, 1)
        activation=nn.ReLU()
        return activation(layer)

def max_pool(x, ksize, stride):
        pool_out = nn.functional.max_pool2d(x, kernel_size=ksize, stride=stride, padding=0,ceil_mode=True)
        return pool_out

def flatten_layer(layer):
        layer_shape = layer.size()
        num_features = layer_shape[1:].numel()
        layer_flat = layer.view(-1, num_features)
        return layer_flat

def fnn_layer(x, w, b):
        A = torch.matmul(x, w) + b
        activation=nn.ReLU()
        return activation(A)

def Branch_b(X, W, b):
    inputs = X  # Assuming Xmin and Xmax scaling is not required in PyTorch
    L = len(W)
    for i in range(L-1):
        outputs = torch.matmul(inputs, W[i]) + b[i]
        activation=nn.ReLU()
        inputs = activation(outputs)  # inputs to the next layer
    Y = torch.matmul(inputs, W[-1]) + b[-1]
    return Y

def fnn_T(X, W, b):
    inputs = X  # Assuming Xmin and Xmax scaling is not required in PyTorch
    L = len(W)
    for i in range(L-1):
        outputs = torch.matmul(inputs, W[i]) + b[i]
        inputs = torch.tanh(outputs)  # inputs to the next layer
    Y = torch.matmul(inputs, W[-1]) + b[-1]
    Y = torch.tanh(Y) 
    return Y

#output dimension for Branch and Trunk Net
num_cs = 7 
G_dim = 50
# #input dimension for Branch Net2
u_dim = G_dim*4
# learning rate
lr = 1e-4 
#Branch Net
layers_br_B = [u_dim] + [32]*1 + [G_dim]
#n_out_channels = 16
filter_size_1 = 3
filter_size_2 = 3
filter_size_3 = 3
filter_size_4 = 3
filter_size_5 = 3
stride = 1

#filter size for each convolutional layer
num_filters_1 = 1
num_filters_2 = 32
num_filters_3 = 64
num_filters_4 = 128
num_filters_5 = 128
num_filters_6 = 128


def predict_qr(paramsb, paramst, R1inv, R2inv, R3inv, data):
    phase, strain_2=data
    W_trunk = paramst[3:7]
    b_trunk = paramst[7:11]
    w1 = paramsb[0]
    b1 = paramsb[1]
    w2 = paramsb[2]
    b2 = paramsb[3]
    w3 = paramsb[4]
    b3 = paramsb[5]
    w4 = paramsb[6]
    b4 = paramsb[7]
    w5 = paramsb[8]
    b5 = paramsb[9]
    w6 = paramsb[10]
    b6 = paramsb[11]
    W_branch_B1 = paramsb[12:14]
    b_branch_B1 = paramsb[14:16]
    W_branch_B2 = paramsb[16:18]
    b_branch_B2 = paramsb[18:20]
    W_branch_B3 = paramsb[20:22]
    b_branch_B3 = paramsb[22:24]
    t = fnn_T(strain_2, W_trunk, b_trunk)
    t1, t2, t3 = t[:,:G_dim], t[:,G_dim:2*G_dim], t[:,G_dim*2:]
    t1_new = torch.matmul(t1, R1inv) #*strain.T    # (size_batch, num_x)
    t2_new = torch.matmul(t2, R2inv) #*strain.T    # (size_batch, num_x)
    t3_new = torch.matmul(t3, R3inv) #*strain.T    # (size_batch, num_x)

    num_samples = phase.shape[0]
    #phase = totorch(phase.reshape([phase.shape[0], phase.shape[1], phase.shape[2], -1]).permute((0,3,1,2)))
    phase = phase.reshape([phase.shape[0]*phase.shape[1],1,phase.shape[2],phase.shape[3]])
    conv_1 = conv_layer(phase, filter_size_1, stride, 1, w1, b1)
    pool_1 = max_pool(conv_1, ksize=3, stride=2)
    conv_2 = conv_layer(pool_1, filter_size_2, stride,1, w2, b2)
    pool_2 = max_pool(conv_2, ksize=3, stride=2)
    conv_3 = conv_layer(pool_2, filter_size_3, stride, 1,w3, b3)
    pool_3 = max_pool(conv_3, ksize=3, stride=2)
    conv_4 = conv_layer(pool_3, filter_size_4, stride, 1,w4, b4)
    pool_4 = max_pool(conv_4, ksize=3, stride=2)
    conv_5 = conv_layer(pool_4, filter_size_5, stride, 0,w5, b5)
    layer_flat = flatten_layer(conv_5)
    emb = fnn_layer(layer_flat, w6, b6)
    emb = emb.reshape([num_samples, num_cs, 3, emb.shape[-1]])
    emb_dir_wise = emb.max(dim=1)[0]
    emb_max = emb_dir_wise.max(dim=1)[0]
    emb_min = emb_dir_wise.min(dim=1)[0]
    emb_mid = emb_dir_wise.sum(dim=1)-emb_max-emb_min
    emb_structure = torch.concat([emb_max, emb_mid, emb_min],dim=1)
    emb_d1 = torch.concat([emb_structure, emb_dir_wise[:,0]],dim=1)
    emb_d2 = torch.concat([emb_structure, emb_dir_wise[:,1]],dim=1)
    emb_d3 = torch.concat([emb_structure, emb_dir_wise[:,2]],dim=1)
    br1 = Branch_b(emb_d1, W_branch_B1, b_branch_B1)
    br2 = Branch_b(emb_d2, W_branch_B2, b_branch_B2)
    br3 = Branch_b(emb_d3, W_branch_B3, b_branch_B3)

    stress_1_pred = torch.matmul(br1, t1_new.T)
    stress_2_pred = torch.matmul(br2, t2_new.T)
    stress_3_pred = torch.matmul(br3, t3_new.T)
    stress_pred = torch.concat([stress_1_pred[:,None], stress_2_pred[:,None], stress_3_pred[:,None]], dim=1)
    stress_pred = stress_pred.reshape([-1, 3, NUM_STAGES, int(round(stress_pred.shape[-1]/NUM_STAGES))])

    return stress_pred

# ############ IMPORTANT #############
# def forward_QR_DeepONet(phase_2d_pt, model_path):

#     # Because the model parameters are saved separately for b and t and R1-3,
#     # model_path becomes a list (or a tuple) which has five elements,
#     # Unpack this model_path list/tuple
#     filename_b, filename_t, filename_R1, filename_R2, filename_R3 = model_path

#     paramsb = load_model(filename_b)
#     paramst = load_model(filename_t)
#     R1 = load_model(filename_R1)
#     R2 = load_model(filename_R2)
#     R3 = load_model(filename_R3)
#     R1inv=torch.linalg.inv(R1)
#     R2inv=torch.linalg.inv(R2)
#     R3inv=torch.linalg.inv(R3)

#     stages = np.linspace(-1,1,NUM_STAGES)
#     strain_all, stages_all = np.meshgrid(strain,stages,indexing='xy')
#     strain_all, stages_all = strain_all.reshape([-1,1]), stages_all.reshape([-1,1])
#     strain_all, stages_all = totorch(strain_all), totorch(stages_all)
#     strain_2 = torch.concat((strain_all, stages_all),dim=1)

#     stress_pool_batch_pred = predict(paramsb, paramst, R1inv, R2inv, R3inv, [phase_2d_pt, strain_2])*NORMALIZER
#     stress_pool_batch_pred = tonumpy(stress_pool_batch_pred)

#     return 
########################################
