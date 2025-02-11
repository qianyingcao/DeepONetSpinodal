import os
import torch
import time
import math
import numpy as np
import sys

from config import *

def tooh(x):
    if x.dtype == 'float': x = x.astype('int')
    outp = np.zeros((x.size, x.max()+1))
    outp[np.arange(x.size),x] = 1
    return outp

def fromoh(x):
    if x.ndim == 2: return np.argmax(x,axis=1)
    if x.ndim == 1: return np.argmax(x)

def tonumpy(*x):
    # x: one tensor, or one list with multiple tensors, or one tuple with multiple tensors, or multiple tensors
    if len(x)==1:
        if torch.is_tensor(x[0]): return x[0].cpu().detach().numpy()
        if isinstance(x[0],list): return [tonumpy(i) for i in x[0]]
        if isinstance(x[0],tuple): return [tonumpy(i) for i in x[0]]
    else:
        return [tonumpy(i) for i in x]

def totorch(x):
    return torch.tensor(x,dtype=torch.float,device=DEVICE)

def batch_info(num_train):
    num_batch = int(np.ceil(num_train/BATCH_SIZE))
    dat = np.zeros((num_batch,3),dtype=int)      # start, end, length
    dat[:,0] = np.arange(0,num_train,BATCH_SIZE)
    dat[:,1] = dat[:,0] + BATCH_SIZE
    dat[:,2] = BATCH_SIZE
    dat[-1,1] = num_train
    dat[-1,2] = num_train-dat[-1,0]
    return dat, dat.shape[0]