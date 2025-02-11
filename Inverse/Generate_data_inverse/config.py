import numpy as np
import sys
import torch
import argparse
import copy


np.set_printoptions(threshold=sys.maxsize)

# # Random Seeds
# SEED = 123
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# Device
if torch.cuda.is_available():  
    dev = "cuda:0" 
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"  
DEVICE = torch.device(dev)

BATCH_SIZE = 512
NUM_STAGES = 2

CASE_NAME0 = 'onlycase0'
CASE_NAME = 'onlycase'

NORMALIZER = 50

ANGLE_TARGET = [[0, 22.5, 30],  #[0, 15, 30], [0, 30 ,30], 1:1
                [0, 15  , 35],  #[0, 15, 15], [0, 15 ,45], 1:2
                [0,  0  , 22.5],#[0,  0, 15], [0,  0 ,30], 1:1
                [0,  0  , 22.5],#[0,  0, 15], [0,  0 ,30], 1:1
                [67.5,  0 ,  0],  #[60, 0,  0], [75, 0 , 0], 1:1
                [52.5, 52.5, 52.5],  #[45, 45,  45], [60, 60 , 60], 1:1, isotropic, two x average
                [0,   0 ,37.5]] #[0,  0, 30], [0,  0 ,45], 1:1
                

