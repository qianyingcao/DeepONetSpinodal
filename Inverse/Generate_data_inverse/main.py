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


from func import *
from config import *
import pickle
import data_inv

os.makedirs('./{}/Fig/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Model/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Output/'.format(CASE_NAME),exist_ok=True)

import pkg_resources
installed_packages = [pkg.project_name for pkg in pkg_resources.working_set]
print("openpyxl" in installed_packages)

data = data_inv.read_data()
strain, stress_out_targets, phase_pool_all, angle_pool_all, sample_id_pool_all = data
torch.save(data,'data')


