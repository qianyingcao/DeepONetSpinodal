import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import torch.nn.init as init
import torch.optim as optim
import pickle

#from Trunk_net import *
from func import *
from config import *
from plot import *

# Import Data
os.makedirs('./model_r', exist_ok=True)

data = io.loadmat('./data2.mat')
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

#strain = strain * (2.0/0.3) - 1.0
stages = np.linspace(-1,1,NUM_STAGES)
strain_all, stages_all = np.meshgrid(strain,stages,indexing='xy')
#strain_all[::2] = strain_all[::2, ::-1]
strain_all, stages_all = strain_all.reshape([-1,1]), stages_all.reshape([-1,1])
strain_all, stages_all = totorch(strain_all), totorch(stages_all)
strain_2 = torch.concat((strain_all, stages_all),dim=1)


def save_model(param_batch1, param_batch2, n):
    filename = './model_r/model' + '.' + str(n)
    with open(filename, 'wb') as file:
        pickle.dump(param_batch1, file)
        pickle.dump(param_batch2, file)

def hyper_initial_WB(layers):
    L = len(layers)
    W = []
    b = []
    for l in range(1, L):
        in_dim = layers[l-1]
        out_dim = layers[l]
        std = np.sqrt(2.0/(in_dim + out_dim))
        weight = init.normal_(torch.empty((in_dim, out_dim), dtype=torch.float32, device='cuda',requires_grad=True), mean=0, std=std)
        bias = init.normal_(torch.empty((1, out_dim), dtype=torch.float32, device='cuda',requires_grad=True), mean=0, std=std)
        W.append(weight)
        b.append(bias)
    return W, b

def matrix_init(N, K):
    in_dim = N
    out_dim = K
    std = np.sqrt(2.0/(in_dim + out_dim))
    W = init.normal_(torch.empty((in_dim, out_dim), dtype=torch.float32, device='cuda',requires_grad=True), mean=0, std=std)
    return W

def fnn_T(X, W, b):
    inputs = X  # Assuming Xmin and Xmax scaling is not required in PyTorch
    L = len(W)
    for i in range(L-1):
        outputs = torch.matmul(inputs, W[i]) + b[i]
        inputs = torch.tanh(outputs)  # inputs to the next layer
    Y = torch.matmul(inputs, W[-1]) + b[-1]
    Y = torch.tanh(Y) 
    return Y

# Trunk dim
x_dim = 2
#output dimension for Branch and Trunk Net
G_dim = 50
# learning rate
lr = 1e-4 
# Trunk Net
layers_tr = [x_dim] + [G_dim]*3 + [G_dim*3]

W_trunk, b_trunk = hyper_initial_WB(layers_tr)
Am11 = matrix_init(G_dim, BATCH_SIZE)
Am12 = matrix_init(G_dim, BATCH_SIZE)
Am21 = matrix_init(G_dim, BATCH_SIZE)
Am22 = matrix_init(G_dim, BATCH_SIZE)
Am31 = matrix_init(G_dim, BATCH_SIZE)
Am32 = matrix_init(G_dim, BATCH_SIZE)

def predict(params, data):
    Am1 = params[0]
    Am2 = params[1]
    Am3 = params[2]
    W_trunk = params[3:7]
    b_trunk = params[7:11]
    v, x = data
    t = fnn_T(x, W_trunk, b_trunk)
    t1, t2, t3 = t[:,:G_dim], t[:,G_dim:2*G_dim], t[:,G_dim*2:]
    stress_1_pred = torch.matmul(Am1.T, t1.T)
    stress_2_pred = torch.matmul(Am2.T, t2.T)
    stress_3_pred = torch.matmul(Am3.T, t3.T)
    stress_pred = torch.concat([stress_1_pred[:,None], stress_2_pred[:,None], stress_3_pred[:,None]], dim=1)
    stress_pred = stress_pred.reshape([-1, 3, NUM_STAGES, int(round(stress_pred.shape[-1]/NUM_STAGES))])
    stress_pred = stress_pred * totorch(strain[None,None,None,:])
    return stress_pred

def loss(params, data, u, weight):
    u_preds = predict(params, data)
    loss_data = torch.mean(weight*(u_preds - u)**2)
    mse = loss_data
    return mse

def update(params, data, u, weight):
    """ Compute the gradient for a batch and update the parameters """
    optimizer = optim.Adam(params, lr=lr)
    optimizer.zero_grad()
    loss_value = loss(params, data, u, weight)  # Assuming you have a loss function defined
    loss_value.backward()
    optimizer.step()
    return params, loss_value

# PyTorch equivalent optimizer initialization
params_batch1 = [Am11] + [Am21] + [Am31] + W_trunk + b_trunk
params_batch2 = [Am12] + [Am22] + [Am32] + W_trunk + b_trunk


train_loss, test_loss = [], []
epo = []
start_time = time.time()
n = 0
min_loss = 1000
stress_train = totorch(stress_train)
stress_weight_train = totorch(stress_weight_train)
for epoch in range(NUM_EPOCHS):
    loss_val = 0
    index = 0
    for i in range(0, phase_train.shape[0], BATCH_SIZE):
        if i == 0:
            params_batch1, loss_batch = update(params_batch1, [phase_train[i:i+BATCH_SIZE], strain_2], stress_train[i:i+BATCH_SIZE], stress_weight_train[i:i+BATCH_SIZE])
        else:
            params_batch2, loss_batch = update(params_batch2, [phase_train[i:i+BATCH_SIZE], strain_2], stress_train[i:i+BATCH_SIZE], stress_weight_train[i:i+BATCH_SIZE])
        loss_val += loss_batch.item()
        index += 1
    
    loss_val /= index
    
    
    # if epoch % 100 == 0:
    #     save_model(params, n)
    #     n += 1
    if loss_val < min_loss:
        save_model(params_batch1, params_batch2, n)
        min_loss = loss_val

    epoch_time = time.time() - start_time
    print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} ".format(epoch, epoch_time, loss_val))
    epo.append(epoch)
    train_loss.append(loss_val)
    start_time = time.time()


filename = './model_r/loss'
with open(filename, 'wb') as file:
    pickle.dump((epo,train_loss), file)


epo = np.array(epo)
loss = np.array(train_loss)

fig1, ax1 = plt.subplots(1,1)
fig1.set_figwidth(17)
fig1.set_figheight(10)
ax1.plot(epo,loss,'b',label="Train loss of trunk")
# ax1.fill_between(epo.flatten(),loss-loss_std,loss+loss_std,color='C0')
ax1.set_yscale('log')
plt.savefig("./model_r/loss_trunk.png", dpi=300)
plt.show()