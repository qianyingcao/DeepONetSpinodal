import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy
from plotting import *
import sys
from torch.nn.modules.utils import _quadruple
from Adam import Adam
from scipy.interpolate import griddata
from scipy import interpolate
from scipy import io
#from torchsummary import summary

seed = 23
np.random.seed(seed)
torch.manual_seed(seed)

print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")


CASE_NAME = 'onlycase1'
os.makedirs('./{}/Fig/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Model/'.format(CASE_NAME),exist_ok=True)
os.makedirs('./{}/Output/'.format(CASE_NAME),exist_ok=True)

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

        # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
#        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(6, self.width)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

        self.final_lin_0 = nn.Sequential(nn.AdaptiveAvgPool2d((4, 52)),

                         nn.Linear(52,51))#,

                         #nn.ReLU())#,

        self.final_lin_1 = nn.Linear(4,2)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
#        x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

#        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 4, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        ########Equivariance-preservation unit###############
        emb_dir_wise = x.max(dim=3)[0]
        emb_max0 = emb_dir_wise.max(dim=3)[0]
        emb_min0 = emb_dir_wise.min(dim=3)[0]
        emb_mid0 = emb_dir_wise.sum(dim=3)-emb_max0-emb_min0
        emb_max = emb_max0.reshape([x.shape[0],-1])
        emb_min = emb_min0.reshape([x.shape[0],-1])
        emb_mid = emb_mid0.reshape([x.shape[0],-1])
        emb_structure = torch.concat([emb_max, emb_mid, emb_min],dim=1)

        emb_d1 = torch.concat([emb_structure, emb_dir_wise[:,:,:,0].reshape([x.shape[0],-1])],dim=1).reshape([x.shape[0],1,102,102])
        emb_d2 = torch.concat([emb_structure, emb_dir_wise[:,:,:,1].reshape([x.shape[0],-1])],dim=1).reshape([x.shape[0],1,102,102])
        emb_d3 = torch.concat([emb_structure, emb_dir_wise[:,:,:,2].reshape([x.shape[0],-1])],dim=1).reshape([x.shape[0],1,102,102])
        #####################################################
         
        x01 = self.final_lin_0(emb_d1)
        x1 = self.final_lin_1(x01.permute(0,1,3,2)).permute(0,1,3,2)
        x02 = self.final_lin_0(emb_d2)
        x2 = self.final_lin_1(x02.permute(0,1,3,2)).permute(0,1,3,2)
        x03 = self.final_lin_0(emb_d3)
        x3 = self.final_lin_1(x03.permute(0,1,3,2)).permute(0,1,3,2)
        x = torch.concat([x1, x2, x3],dim=1)

        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)

################################################################
# configs
################################################################
def FNO_main(save_index):
    
    data = io.loadmat('./data1.mat')
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
    num_train = phase_train.shape[0]
    num_test = phase_test.shape[0]

    x_train = torch.tensor(phase_train)
    y_train = torch.tensor(stress_train)
    x_test = torch.tensor(phase_test)
    y_test = torch.tensor(stress_test)

    batch_size_train = 16
    batch_size_test = 16
    epochs = 1000
    learning_rate = 0.001
    step_size = 100
    gamma = 0.5

    modes1 = 25
    modes2 = 25
    modes3 = 4
    width = 16
        
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, torch.tensor(stress_weight_train)), batch_size=batch_size_train, shuffle=True)
    train_loaderL2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, torch.tensor(stress_weight_train)), batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, torch.tensor(stress_weight_test)), batch_size=batch_size_test, shuffle=True)

    device = torch.device('cuda')    
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO3d(modes1, modes2, modes3, width).cuda()
    # num_parameters = count_params(model)
    # print("Number of trainable parameters: %d" %(num_parameters))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start_time = time.time()

    myloss = LpLoss(size_average=True)       
    
    train_rela_l2 = np.zeros((epochs, 1))
    test_rela_l2 = np.zeros((epochs, 1))
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        for x, y, weight in train_loader:
            x, y, weight = x.cuda(), y.cuda(), weight.cuda()     
            optimizer.zero_grad()
            out1 = model(x.float())
            strain_new = torch.tensor(strain[None,None,None,:])
            out = out1 * strain_new.cuda()
            #loss = myloss(out, y)
            MSE = torch.mean(weight*(out - y)**2)
            loss = torch.sqrt(MSE)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()  
    
        scheduler.step()    
        model.eval()
        train_mse = 0.0

        with torch.no_grad():
            for x, y, weight in train_loaderL2:
                x, y, weight = x.cuda(), y.cuda(), weight.cuda()    
                out1 = model(x.float())
                out = out1 * strain_new.cuda()
                train_mse +=  torch.mean(weight*(out - y)**2)
        test_mse = 0
        with torch.no_grad():
            for x, y, weight in test_loader:
                x, y, weight = x.cuda(), y.cuda(), weight.cuda()    
                out1 = model(x.float())
                out = out1 * strain_new.cuda()
                test_mse +=  torch.mean(weight*(out - y)**2)
                
        train_loss /= num_train
        train_mse /= num_train
        test_mse /= num_test

        train_rela_l2[ep,0] = train_mse

        test_rela_l2[ep,0] = test_mse


        t2 = default_timer()
        
        print("Epoch: %d, time: %.7f, Train Loss: %.7e, Train mse: %.7f, Test mse: %.7f" % ( ep, t2-t1, train_loss, train_mse, test_mse))
   
    elapsed = time.time() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")
    #summary(model,(39,14,14,4),device="cuda")


    # ====================================
    # saving settings
    # ====================================
    current_directory = os.getcwd()
    case = "Case_"
    folder_index = str(save_index)
    
    results_dir = "/" + case + folder_index +"/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)
    
    np.savetxt(save_results_to+'/train_loss.txt', train_rela_l2)
    np.savetxt(save_results_to+'/test_loss.txt', test_rela_l2)
        
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)
        
    torch.save(model, save_models_to+'Model.pt')


    ################################################################
    # testing
    ################################################################
    
    #dump_test =  save_results_to+'/Predictions/Test/'
    #os.makedirs(dump_test, exist_ok=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, torch.tensor(stress_weight_test)), batch_size=1, shuffle=False)    
    pred_u = torch.zeros(num_test,3,2,51)

    index = 0
    MSE = 0
    t1 = default_timer()
    with torch.no_grad():
        for x, y, weight in test_loader:
            
            x, y, weight = x.cuda(), y.cuda(), weight.cuda()    
            out1 = model(x.float()) 
            out = out1 * strain_new.cuda()
            MSE += torch.mean(weight*(out - y)**2)
            pred_u[index,:,:,:] = out
            index = index + 1

    MSE = MSE/index
    t2 = default_timer()
    testing_time = t2-t1

    saved = {}
    saved['data'] = (strain, stress_test, pred_u.cpu().numpy(), stress_weight_test, angle_test)
    np.save('./{}/Output/SavedOutputs_final.npy'.format(CASE_NAME), saved)

    scipy.io.savemat(save_results_to+'pred_test_data.mat', 
                      mdict={'stress_test': stress_test, 
                             'stress_pred': pred_u.cpu().numpy(),
                             'MSE': MSE.cpu().numpy(),
                             'Train_time':elapsed,
                             'Test_time':testing_time})  
    
    print("\n=============================")
    print('Mean squared error: %.3e'%(MSE))
    print("=============================\n")    


if __name__ == "__main__":
    
    save_index = 1
    FNO_main(save_index)
