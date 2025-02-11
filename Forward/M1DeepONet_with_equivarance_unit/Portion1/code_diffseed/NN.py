import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from func import *


class NN(nn.Module):
    def __init__(self, strain, num_cs):
        super(NN, self).__init__()

        self.strain = strain

        self.num_cs = num_cs

        self.p = 30
        self.emb_size = 30

        self.branch_A = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, device=DEVICE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=DEVICE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, device=DEVICE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, device=DEVICE),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, device=DEVICE),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(128, self.emb_size, device=DEVICE),
            nn.ReLU(inplace=True)
        )

        self.branch_B = nn.Sequential(
            nn.Linear(self.emb_size*4, 32, device=DEVICE),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.p, device=DEVICE)
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(1+1, 32, device=DEVICE),      # strain + stage
            nn.Tanh(),
            nn.Linear(32, 32, device=DEVICE),
            nn.Tanh(),
            nn.Linear(32, 32, device=DEVICE),
            nn.Tanh(),
            nn.Linear(32, self.p*3, device=DEVICE),
            nn.Tanh()
        )

        # mechanics - b.*t bias
        self.NN_bias = nn.Parameter(torch.tensor(np.zeros((3,)), dtype=torch.float, requires_grad=True, device=DEVICE))

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)

        # LR scheduler
        decay_rate = 0.5**(1/5000)          # after 4K epochs (see main), halve the lr every 10K epochs
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, decay_rate)
    
    def _forward(self, phase):

        strain = self.strain * (2.0/0.3) - 1.0
        stages = np.linspace(-1,1,NUM_STAGES)
        strain_all, stages_all = np.meshgrid(strain,stages,indexing='xy')
        #strain_all[::2] = strain_all[::2, ::-1]
        strain_all, stages_all = strain_all.reshape([-1,1]), stages_all.reshape([-1,1])
        strain_all, stages_all = totorch(strain_all), totorch(stages_all)

        num_samples = phase.shape[0]
        phase = phase.reshape([phase.shape[0]*phase.shape[1],1,phase.shape[2],phase.shape[3]])
        emb = self.branch_A(phase)
        emb = emb.reshape([num_samples, self.num_cs, 3, emb.shape[-1]])
        emb_dir_wise = emb.max(dim=1)[0]
        emb_max = emb_dir_wise.max(dim=1)[0]
        emb_min = emb_dir_wise.min(dim=1)[0]
        emb_mid = emb_dir_wise.sum(dim=1)-emb_max-emb_min
        emb_structure = torch.concat([emb_max, emb_mid, emb_min],dim=1)
        emb_d1 = torch.concat([emb_structure, emb_dir_wise[:,0]],dim=1)
        emb_d2 = torch.concat([emb_structure, emb_dir_wise[:,1]],dim=1)
        emb_d3 = torch.concat([emb_structure, emb_dir_wise[:,2]],dim=1)
        b1 = self.branch_B(emb_d1)
        b2 = self.branch_B(emb_d2)
        b3 = self.branch_B(emb_d3)
        t = self.trunk_net(torch.concat((strain_all, stages_all),dim=1))
        t1, t2, t3 = t[:,:self.p], t[:,self.p:2*self.p], t[:,self.p*2:]
        stress_1_pred = (torch.matmul(b1,t1.T)+self.NN_bias[0]) #*strain.T    # (size_batch, num_x)
        stress_2_pred = (torch.matmul(b2,t2.T)+self.NN_bias[1]) #*strain.T    # (size_batch, num_x)
        stress_3_pred = (torch.matmul(b3,t3.T)+self.NN_bias[2]) #*strain.T    # (size_batch, num_x)
        stress_pred = torch.concat([stress_1_pred[:,None], stress_2_pred[:,None], stress_3_pred[:,None]], dim=1)
        stress_pred = stress_pred.reshape([-1, 3, NUM_STAGES, int(round(stress_pred.shape[-1]/NUM_STAGES))])
        stress_pred = stress_pred * totorch(self.strain[None,None,None,:])

        return stress_pred, t1, t2, t3
    

    def _loss(self, output_pred, output_target, weight):
        return torch.mean(weight*(output_pred - output_target)**2)

    def preprocess_input(self, X, y, y_weight):
        X = totorch(X.reshape([X.shape[0], X.shape[1], X.shape[2], -1]).transpose((0,3,1,2)))
        y = totorch(y)
        y_weight = totorch(y_weight)
        return X, y, y_weight

    def iterate_once(self, iepoch, X_train, y_train, y_weight_train):


        X_train, y_train, y_weight_train = self.preprocess_input(X_train, y_train, y_weight_train)
        tic = time.time()
        num_train = X_train.shape[0]
        batches, num_batches = batch_info(num_train)
        perm = np.random.permutation(num_train)

        def fw(choice=None):

            # Batch
            if choice is None: choice = np.arange(num_train)
            X_batch = X_train[choice]
            y_batch = y_train[choice]
            y_weight_batch = y_weight_train[choice]

            # Forward
            y_pred_batch, t1, t2, t3 = self._forward(X_batch)

            # Loss
            loss_batch = self._loss(y_pred_batch, y_batch, y_weight_batch)

            return loss_batch, y_pred_batch, t1, t2, t3
        
        # Batch optimization
        for ibatch in range(num_batches):
            istart = batches[ibatch,0]
            iend = batches[ibatch,1]
            choice = perm[istart:iend]
            self.optimizer.zero_grad()
            loss_batch, y_pred, t1, t2, t3 = fw(choice)
            loss_batch.backward()
            if iepoch>=0: self.optimizer.step()
            
        if iepoch>=0: self.scheduler.step()

        # A final forward run for outputing
        #loss, y_pred, t1, t2, t3 = fw()
        
        # Timing
        toc = time.time()
        dt = toc-tic

        # Returning Numpy Values
        loss_np = tonumpy(loss_batch)
        y_pred = tonumpy(y_pred)
        # y_weight_train = tonumpy(y_weight_train)
        # y_pred = np.where(y_weight_train==0.0, np.nan, y_pred)
        
        return loss_np, y_pred, dt, t1, t2, t3

    def test(self, X, y, y_weight):
        X, y, y_weight = self.preprocess_input(X, y, y_weight)
        y_pred, t1, t2, t3 = self._forward(X)
        loss = self._loss(y_pred, y, y_weight)
        y_pred, loss = tonumpy(y_pred), tonumpy(loss)
        y_weight = tonumpy(y_weight)
        y_pred = np.where(y_weight==0.0, np.nan, y_pred)
        return loss, y_pred, t1, t2, t3