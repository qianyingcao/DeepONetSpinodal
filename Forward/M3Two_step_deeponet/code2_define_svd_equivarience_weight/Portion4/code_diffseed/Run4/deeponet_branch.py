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


from func import *
from config import *
from plot import *


os.makedirs('./model_b', exist_ok=True)
model = "0"

# Import Data
data = io.loadmat('./data4.mat')
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
strain_all, stages_all = strain_all.reshape([-1,1]), stages_all.reshape([-1,1])
strain_all, stages_all = totorch(strain_all), totorch(stages_all)
strain_2 = torch.concat((strain_all, stages_all),dim=1)

def save_model(param, n):
    filename = './model_b/model' + '.' + str(n)
    with open(filename, 'wb') as file:
        pickle.dump(param, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        param=pickle.load(file)
    return param

def hyper_initial(shape_w, shape_b):
        std = 0.01
        weight = nn.Parameter(torch.randn(shape_w, dtype=torch.float32, device='cuda') * std, requires_grad=True)
        bias = nn.Parameter(torch.zeros(shape_b, dtype=torch.float32, device='cuda'), requires_grad=True)
        return weight, bias

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


def hyper_initial_linear(shape_w, shape_b):
        std = 0.01
        weight = nn.Parameter(torch.randn(shape_w, dtype=torch.float32, device='cuda') * std, requires_grad=True)
        bias = nn.Parameter(torch.randn(shape_b, dtype=torch.float32, device='cuda'), requires_grad=True)
        return weight, bias

def fnn_layer(X, W, b):
        A = torch.matmul(X, W) + b
        activation=nn.ReLU()
        return activation(A)

def hyper_initial_Branch_b(layers):
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
G_dim = 50
# #input dimension for Branch Net2
u_dim = G_dim*4
# learning rate
lr = 1e-4 
#Branch Net
layers_br_B = [u_dim] + [32]*1 + [G_dim]
print("branch layers:\t",layers_br_B)

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

shape_w1 = [num_filters_2, num_filters_1, filter_size_1, filter_size_1]
shape_b1 = [num_filters_2]
shape_w2 = [num_filters_3, num_filters_2, filter_size_2, filter_size_2]
shape_b2 = [num_filters_3]
shape_w3 = [num_filters_4, num_filters_3, filter_size_3, filter_size_3]
shape_b3 = [num_filters_4]
shape_w4 = [num_filters_5, num_filters_4, filter_size_4, filter_size_4]
shape_b4 = [num_filters_5]
shape_w5 = [num_filters_6, num_filters_5, filter_size_5, filter_size_5]
shape_b5 = [num_filters_6]
shape_w6 = [num_filters_6, G_dim]
shape_b6 = [1,G_dim]
w1, b1 = hyper_initial(shape_w1, shape_b1)
w2, b2 = hyper_initial(shape_w2, shape_b2)
w3, b3 = hyper_initial(shape_w3, shape_b3)
w4, b4 = hyper_initial(shape_w4, shape_b4)
w5, b5 = hyper_initial(shape_w5, shape_b5)
w6, b6 = hyper_initial_linear(shape_w6, shape_b6)

W_branch_B1, b_branch_B1 = hyper_initial_Branch_b(layers_br_B)
W_branch_B2, b_branch_B2 = hyper_initial_Branch_b(layers_br_B)
W_branch_B3, b_branch_B3 = hyper_initial_Branch_b(layers_br_B)


def predict1(params, data):
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
    return t1, t2, t3, Am1, Am2, Am3

filename = "./model_r/"+"model"+"."+model
params = load_model(filename)

phi1, phi2, phi3, Am1, Am2, Am3 = predict1(params,[phase_train, strain_2])
U1,S1,V1 = torch.linalg.svd(phi1,full_matrices=False)
U2,S2,V2 = torch.linalg.svd(phi2,full_matrices=False)
U3,S3,V3 = torch.linalg.svd(phi3,full_matrices=False)
################################################
phi = torch.concat((phi1, phi2, phi3),dim=1)
U,S,V = torch.linalg.svd(phi,full_matrices=False)
save_results_to = './basis/'
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)
io.savemat(save_results_to+'Basis_two_step_SVD_unit.mat',
            mdict={ 'U1': U1.detach().cpu().numpy(),
                    'U2': U2.detach().cpu().numpy(),
                    'U3': U3.detach().cpu().numpy(),
                    'U': U.detach().cpu().numpy(),
                    'S1': S1.detach().cpu().numpy(),
                    'S': S1.detach().cpu().numpy()})
################################################
R1 = torch.matmul(torch.diag(S1),V1)
R2 = torch.matmul(torch.diag(S2),V2)
R3 = torch.matmul(torch.diag(S3),V3)
u_train1=torch.einsum('ij,jk->ki', R1, Am1)
u_train2=torch.einsum('ij,jk->ki', R2, Am2)
u_train3=torch.einsum('ij,jk->ki', R3, Am3)
u_train = torch.concat([u_train1, u_train2, u_train3], dim=1)
print("phi1=\t",phi1.shape)
print("Am1=\t",Am1.shape)
print("R1=\t",R1.shape)
print("u_train1=\t",u_train1.shape)
print("u_train=\t",u_train.shape)

def predict(params, phase):
    w1 = params[0]
    b1 = params[1]
    w2 = params[2]
    b2 = params[3]
    w3 = params[4]
    b3 = params[5]
    w4 = params[6]
    b4 = params[7]
    w5 = params[8]
    b5 = params[9]
    w6 = params[10]
    b6 = params[11]
    W_branch_B1 = params[12:14]
    b_branch_B1 = params[14:16]
    W_branch_B2 = params[16:18]
    b_branch_B2 = params[18:20]
    W_branch_B3 = params[20:22]
    b_branch_B3 = params[22:24]
    num_samples = phase.shape[0]
    phase = totorch(phase.reshape([phase.shape[0], phase.shape[1], phase.shape[2], -1]).transpose((0,3,1,2)))
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
    b1 = Branch_b(emb_d1, W_branch_B1, b_branch_B1)
    b2 = Branch_b(emb_d2, W_branch_B2, b_branch_B2)
    b3 = Branch_b(emb_d3, W_branch_B3, b_branch_B3)
    stress_pred = torch.concat([b1, b2, b3], dim=1)
    return stress_pred


def loss(params, phase, u ):
    u_preds = predict(params, phase)
    loss_data = torch.mean((u_preds - u)**2)
    mse = loss_data
    return mse

def update(params, phase, u):
    """ Compute the gradient for a batch and update the parameters """
    optimizer.zero_grad()
    loss_value = loss(params, phase, u)  # Assuming you have a loss function defined
    loss_value.backward()
    optimizer.step()
    return params, loss_value

# PyTorch equivalent optimizer initialization
params =  [w1] + [b1] + [w2] + [b2] + [w3] + [b3] + [w4] + [b4] + [w5] + [b5] + [w6] + [b6] + W_branch_B1 + b_branch_B1 + W_branch_B2 + b_branch_B2 + W_branch_B3 + b_branch_B3
optimizer = optim.Adam(params, lr=lr)

train_loss, test_loss = [], []
epo = []
start_time = time.time()
n = 0
min_loss = 1000
u_train = totorch(u_train)
for epoch in range(NUM_EPOCHS):

    params, loss_val = update(params, phase_train, u_train)
    # if epoch % 100 == 0:
    #     save_model(params, n)
    #     n += 1
    if loss_val < min_loss:
        save_model(params, n)
        min_loss = loss_val

    if epoch % 100 == 0:
        epoch_time = time.time() - start_time
        u_train_pred = predict(params, phase_train)
        err_train = torch.mean(torch.norm(u_train - u_train_pred, dim=1) / torch.norm(u_train, dim=1))
        l1 = loss(params, phase_train, u_train).item()
        train_loss.append(l1)
        #print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} | Train L2: {:0.6f}".format(epoch, epoch_time, l1, err_train))
        print("Epoch {} | T: {:0.6f} | Train MSE: {:0.3e} ".format(epoch, epoch_time, l1))
        epo.append(epoch)
    start_time = time.time()
save_model(params,n)

filename = './model_b/Rmatrix1'
with open(filename, 'wb') as file:
    pickle.dump(R1, file)

filename = './model_b/Rmatrix2'
with open(filename, 'wb') as file:
    pickle.dump(R2, file)

filename = './model_b/Rmatrix3'
with open(filename, 'wb') as file:
    pickle.dump(R3, file)

filename = './model_b/loss'
with open(filename, 'wb') as file:
    pickle.dump((epo,train_loss), file)

pred = predict(params, phase_train)

epo = np.array(epo)
loss = np.array(train_loss)

fig1, ax1 = plt.subplots(1,1)
fig1.set_figwidth(17)
fig1.set_figheight(10)
ax1.plot(epo,loss,'b',label="Train loss of branch")
# ax1.fill_between(epo.flatten(),loss-loss_std,loss+loss_std,color='C0')
ax1.set_yscale('log')
plt.savefig("./model_b/loss_branch.png", dpi=300)
plt.show()