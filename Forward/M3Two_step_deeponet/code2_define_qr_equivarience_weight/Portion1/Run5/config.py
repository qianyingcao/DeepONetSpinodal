import numpy as np
import sys
import torch
import argparse


np.set_printoptions(threshold=sys.maxsize)

# Random Seeds
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)

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
NUM_EPOCHS = 80000
BATCH_SIZE = 1125
PLOT_ITS = [NUM_EPOCHS]
SAVE_ITS = [NUM_EPOCHS]

PCT_TEST = 0.1

#TEST_SET = (0,0,15)

NEGATIVE_NUMBER = -2.0

PORTION = 1  #4

DOWN_SAMPLE = 2

NUM_STAGES = 2

ANGLES_uncertain = [[ 0,0,20],[ 0,0,30], [30, 30, 30], [0, 30, 30], [0, 30, 60], [0, 45, 60], [15, 15, 30]]

# ANGLES_ALL = [[ 0, 0,15],
#               [ 0, 0,20],
#               [ 0, 0,30],
#               [ 0,15,15],
#               [ 0,15,20],
#               [ 0,15,30],
#               [ 0,15,45],
#               [ 0,15,60],
#               [ 0, 20,20],
#               [ 0, 30,30],
#               [ 0, 30,45],
#               [ 0, 30,60],
#               [0, 45, 45],
#               [ 0, 45,60],
#               [15,15,15],
#               [15,15,30],
#               [15,15,45],
#               [15, 30, 45],
#               [30, 30, 30],
#               [30, 30, 45],
#               [45, 0, 0],
#               [60, 0, 0],
#               [90, 0, 0]]

#ANGLES_ALL = set(tuple(item) for item in ANGLES_ALL)

# NUM_IMAGES_PER_SAMPLE = 20

# PERM = np.random.permutation(len(ANGLES_ALL)*NUM_IMAGES_PER_SAMPLE)

# def get_parse_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('code', metavar='K', type=int)
#     args = parser.parse_args()
#     code = args.code
#     ### code = id_test_ssid * num_seeds + id_seed ###
#     id_test = code
#     test_ids = PERM[id_test*NUM_IMAGES_PER_SAMPLE:(id_test+1)*NUM_IMAGES_PER_SAMPLE]
#     print('======== Test_setup: {} ========'.format(id_test))
#     print(test_ids)
#     return test_ids, id_test

# TEST_IDS, id_test = tuple(get_parse_params())

# CASE_NAME = 'testsetup_{}'.format(id_test)

#CASE_NAME = 'onlycase'