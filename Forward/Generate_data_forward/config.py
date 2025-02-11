import numpy as np
import sys
import torch
import argparse


np.set_printoptions(threshold=sys.maxsize)

# Random Seeds

# Device
if torch.cuda.is_available():  
    dev = "cuda:0" 
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"  
DEVICE = torch.device(dev)
print(DEVICE)

# DL setup
#INP_DIM = 2
#OUTP_DIM = 2
NUM_PTS_TEST_1D = 100
NUM_SUBPLOT = 5
NUM_EPOCHS = 10000
BATCH_SIZE = 32
PLOT_ITS = [NUM_EPOCHS]
SAVE_ITS = [NUM_EPOCHS]

PCT_TEST = 0.1

#TEST_SET = (0,0,15)

NEGATIVE_NUMBER = -2.0

PORTION = 1  #4

DOWN_SAMPLE = 2

NUM_STAGES = 2

ANGLES_uncertain = [[ 0,0,20],[ 0,0,30], [0, 30, 30], [30, 30, 30],  [0, 30, 60], [0, 45, 60], [15, 15, 30]]
