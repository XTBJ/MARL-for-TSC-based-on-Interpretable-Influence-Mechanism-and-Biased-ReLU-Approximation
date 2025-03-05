'''
@ Summary: BReLU Neural Network, Pytorch Version
@ Author: XTBJ
'''

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from NN import load_data

torch.autograd.set_detect_anomaly(True)

# hyper-parameters
episode = 5000
LR = 0.01
scale = 0.02

### Incremental Learning
class IncrementalBlock:
    def __init__(self, hidden_size, mean, var, num):
        self.count = 0
        self.mean = torch.zeros([hidden_size]).cuda()
        self.var = torch.ones([hidden_size]).cuda()

    def update(self, x):
        self.count += 1
        x = x.detach().squeeze()
        mean = self.mean.detach()
        var = self.var.detach()
        delta1 = x - mean
        self.mean = mean + delta1/self.count
        delta2 = x - mean
        self.var = (var*(self.count-1) + delta1*delta2)/self.count
        
    def update_batch(self, batch_x):
        for x in batch_x:
            self.update(x)
    def get_mean(self):
        return self.mean
    def get_var(self):
        return self.var


class BReLU_Layer(nn.Module):
    def __init__(self, hidden_size, mean, var, weight):
        super(BReLU_Layer, self).__init__()
        self.IncB = IncrementalBlock(hidden_size, mean, var, num=38000)
        self.hidden_size = hidden_size
        self.biases = Parameter(torch.zeros([hidden_size]))
        self.batch_norm = nn.BatchNorm1d(hidden_size, affine=False)
        self.reset_parameters(weight)
    
    def reset_parameters(self, weight):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.biases is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1/math.sqrt(fan_in) if fan_in>0 else 0
            nn.init.uniform_(self.biases, -bound, bound)

    def forward(self, x, update=False):
        '''
        x: (batch_size, hidden_size)
        normed: (batch_size, hidden_size)
        out: (batch_size, hidden_size*q)
        mean: (hidden_size)
        var: (hidden_size)
        '''
        batch_size = x.shape[0]

        if batch_size > 1:
            x = self.batch_norm(x)
        self.mean = torch.mean(x).detach().squeeze()
        self.var = torch.var(x).detach().squeeze()

        std = torch.sqrt(self.var)
        quantile = [-2.02*std+self.mean,-0.765*std+self.mean,-0.195*std+self.mean,0.315*std+self.mean,0.93*std+self.mean]
       
        out = []
        for i in range(len(quantile)):
            out.append(torch.maximum(torch.tensor(0.0), x-quantile[i]))
        output = torch.cat(out, dim=1)
        return output


# three-layers
class BReLU_Net(nn.Module):
    def __init__(self, in_size, hidden_size1, hidden_size2, out_size, q, mean, var):
        super(BReLU_Net, self).__init__()
        self.w1 = Parameter(torch.randn(in_size, hidden_size1), requires_grad = True)
        self.w2 = Parameter(torch.randn(hidden_size1*(q), hidden_size2), requires_grad = True)
        self.w3 = Parameter(torch.randn(hidden_size2*(q), out_size), requires_grad = True)
        self.batch_norm = nn.BatchNorm1d(in_size, affine=False)
        self.bn1 = BReLU_Layer(hidden_size1, mean, var, self.w1)
        self.bn2 = BReLU_Layer(hidden_size2, mean, var, self.w2)
        self.biases = Parameter(torch.zeros(out_size))
        self.reset_parameters(self.w3)
    
    def reset_parameters(self, weight):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.biases is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1/math.sqrt(fan_in) if fan_in>0 else 0
            nn.init.uniform_(self.biases, -bound, bound)

    def normalization_layer(self, data):
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        normed = (data-data_min)/(data_max-data_min+1e-5)
        return normed

    def forward(self, x):
        x1 = torch.matmul(x, self.w1)
        out1 = self.bn1(x1)

        x2 = torch.matmul(out1, self.w2)
        out2 = self.bn2(x2)

        pre_y = torch.add(torch.matmul(out2, self.w3), self.biases)
        return pre_y, self.w1, self.w2, self.w3
