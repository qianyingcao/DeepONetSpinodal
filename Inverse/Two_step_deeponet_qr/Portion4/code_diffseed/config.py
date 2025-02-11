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
                #[0,  0  , 22.5],#[0,  0, 15], [0,  0 ,30], 1:1
                [0,  0  , 22.5],#[0,  0, 15], [0,  0 ,30], 1:1
                [67.5,  0 ,  0],  #[60, 0,  0], [75, 0 , 0], 1:1
                [52.5, 52.5, 52.5],  #[45, 45,  45], [60, 60 , 60], 1:1, isotropic, two x average
                [0,   0 ,37.5]] #[0,  0, 30], [0,  0 ,45], 1:1
                

# NUM_PTS_TEST_1D = 100
# NUM_SUBPLOT = 5
# NUM_EPOCHS = 10000

# PLOT_ITS = [5,2000,4000,6000,8000,10000]
# SAVE_ITS = PLOT_ITS

# PCT_TEST = 0.1

# #TEST_SET = (0,0,15)

# NEGATIVE_NUMBER = -2.0

# PORTION = 4

# DOWN_SAMPLE = 2



# ANGLES_ALL = [[ 0,15,45],
#               [30,30,30],
#               [45, 0, 0],
#               [ 0,15,60],
#               [15,15,45],
#               [15,15,15],
#               [15,30,30],
#               [ 0,15,30],
#               [90, 0, 0],
#               [ 0,30,30],
#               [ 0,45,45],
#               [ 0,15,15],
#               [15,15,30],
#               [ 0, 0,30],
#               [60, 0, 0],
#               [30,30,45]]
#               #[15,30,45],
#               #[ 0,30,45],
# ANGLES_ALL_NP = np.array(ANGLES_ALL)
# ANGLES_ALL_LIST = copy.deepcopy(ANGLES_ALL)
# ANGLES_ALL = set(tuple(item) for item in ANGLES_ALL)

# CONNECTIVITY = [[7,11],[14,8],[2,14],[1,5],[7,9],[0,3],[9,10],[4,12],[1,15],[0,4],[1,8]]

# ANGLES_ALL_OUT = [[0,15,23],
#                     [75,0,0],
#                     [53,0,0],
#                     [23,23,23],
#                     [0,23,30],
#                     [0,15,53],
#                     [0,38,38],
#                     [15,15,38],
#                     [30,30,38],
#                     [8,15,45],
#                     [45,45,45]]
# ANGLES_ALL_OUT_NP = np.array(ANGLES_ALL_OUT)
# ANGLES_ALL_OUT_LIST = copy.deepcopy(ANGLES_ALL_OUT)
# ANGLES_ALL_OUT = list(tuple(item) for item in ANGLES_ALL_OUT)

# def conn_to_oh(conn):
#     conn = np.array(conn)
#     num_entries = np.max(conn)+1
#     num_rows = conn.shape[0]
#     ans = []
#     for irow in range(num_rows):
#         temp = np.zeros((num_entries,))
#         temp[conn[irow,0]] += 1
#         temp[conn[irow,1]] += 1
#         ans.append(temp)
#     ans = np.array(ans)
#     return ans

# CONNECTIVITY_OH = conn_to_oh(CONNECTIVITY)
# ANGLES_ALL_OUT_PRED = np.ceil(0.5 * CONNECTIVITY_OH @ ANGLES_ALL_NP).astype('int')

# print('Comparison of ANGLES_ALL_OUT_PRED and TRUE:')
# print(ANGLES_ALL_OUT_NP)
# print(ANGLES_ALL_OUT_PRED)
# user_input = input("Press 'Y' to continue or any other key to abort: ")
# if user_input != 'Y':
#     exit()


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

