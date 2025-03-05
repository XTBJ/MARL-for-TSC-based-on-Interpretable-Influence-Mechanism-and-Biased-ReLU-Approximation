'''
@ Summary: EHH + Change ANOVA Decomposition (Using the equation of the origin version)
@ Author: XTBJ
'''

import math
import random
from collections import Counter
import os
import sys

import numpy as np
import statsmodels.api as sm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import init

import multiprocessing as mp
import threading as td
import time
import pandas as pd
from matplotlib import pyplot as plt

# hyperparameters
epsilon = 0.4

class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0-self.decay)*x + self.decay*self.shadow[name]
        self.shadow[name] = new_average.clone()

class Batch_Norm_Layer(nn.Module):
    def __init__(self, hidden_size):
        super(Batch_Norm_Layer, self).__init__()
        self.beta = Parameter(torch.zeros([hidden_size]))
        self.gamma = Parameter(torch.ones([hidden_size]))
        self.ema = EMA(decay=0.5)

    def forward(self, x, num):
        '''
        x: (batch_size, hidden_size)
        mean: (hidden_size)
        var: (hidden_size)
        gamma: (hidden_size)
        beta: (hidden_size)
        '''
        axis = list(range(len(x.shape)-1))
        batch_mean = torch.mean(x, [0])
        batch_var = torch.var(x, [0], unbiased=False)
        data = {"mean"+str(num): batch_mean, "var"+str(num): batch_var}

        for name in data:
            self.ema.register(name, data[name])

        for name in data:
            self.ema.update(name, data[name])
            mean = self.ema.get("mean"+str(num))
            var = self.ema.get("var"+str(num))
        normed = self.gamma*(x-mean)/torch.sqrt(var + torch.tensor(1e-3)) + self.beta
        return normed, self.beta, self.gamma, mean, var

# simple fully connected network
class Encoder(nn.Module):
    # DownSampling
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        # Linear embedding
        embedding = self.fc(x)
        return embedding, self.fc.weight, self.fc.bias

class EHH_Layer(nn.Module):
    def __init__(self, input_size, output_size, intermediate_node=50, quantile=6, file_path='models/adj30.csv'):
        super(EHH_Layer, self).__init__()
        self.input_size = input_size
        self.quantile = quantile
        self.source_node = input_size*quantile
        self.intermediate_node = intermediate_node
        self.output_node = output_size
        self.batch_layer = Batch_Norm_Layer(input_size)
        # test mode
        self.f = pd.read_csv(file_path)
        self.chosen_index = []
        self.w = Parameter(torch.randn(intermediate_node + input_size*quantile, output_size), requires_grad = True)
        self.biases = Parameter(torch.randn(1), requires_grad = True)

    def Maxout_Layer(self, x, num):
        normed, beta, gamma, mean, var = self.batch_layer(x, num)
        output = torch.maximum(torch.tensor(0.0), normed)
        quantile = [-3*var+mean,-0.834*var+mean,-0.248*var+mean,0.248*var+mean,0.834*var+mean]
        sum = []
        sum.append(output)
        for i in range(len(quantile)):
            out = torch.maximum(torch.tensor(0.0), normed-quantile[i]*gamma+beta)
            sum.append(out)
        output = torch.stack(sum, 0)
        return output
    
    def get_epsilon(self, x):
        # Packet operation
        '''
        D: (b, n, q)
        C: (b, n/2, q)
        expanded_C: (b, n, n/2, q)
        expanded_D: (b, n, 1, q)
        Jv: (b, n/2, q)
        '''
        b = x.shape[0]
        n = x.shape[1]
        q = x.shape[2]
        D = x
        D_norm = torch.norm(D, dim=0)
        D = D/(D_norm[None,:] + 1e-5)
        expanded_D = D.unsqueeze(2)
        permutation = torch.randperm(n)
        D1 = x[:, permutation[:int(n/2)], :]
        D2 = x[:, permutation[int(n/2):], :]
        C = torch.min(D1,D2)
        C = C.float()
        C_norm = torch.norm(C, dim=0)
        C = C/(C_norm[None,:] + 1e-5)
        expanded_C = C[:, None, :, :].expand(-1, n, -1, -1)
        Jv_tmp = expanded_C*expanded_D
        Jv, _ = torch.max(Jv_tmp, dim=1)
        Jv_sorted, _ = torch.sort(Jv.view(-1))
        epsilon = Jv_sorted[:self.intermediate_node]
        return epsilon.item()[-1]


    def Minout_Layer(self, x, init_struct=False):
        # define the link between source nodes and intermediate nodes.
        rows = self.source_node
        cols = self.intermediate_node
        n = x.size()[1]
        q = x.size()[2]
        m = self.intermediate_node

        # initialize the EHH structure
        if init_struct:
            # Jv
            out = []
            # Compute the alternative neuron set
            # Only use the information of the source nodes
            self.chosen_index = []
            alter_neurons = {}
            x_after = torch.flatten(x, start_dim=1)

            index_set = set()
            count = 0
            while True:
                n1_index = random.randint(0, n-1)
                n2_index = random.randint(0, n-1)
                while n2_index == n1_index:
                    n2_index = random.randint(0, n-1)
                q1_index = random.randint(0, q-1)
                q2_index = random.randint(0, q-1)
                index = (n1_index, q1_index, n2_index, q2_index)
                while index not in index_set:
                    index_set.add(index)
                    D1 = x[:, n1_index, q1_index]
                    D2 = x[:, n2_index, q2_index]
                    D = torch.stack([D1, D2], 1)
                    v = torch.min(D,1)[0]
                    Jv = []
                    for k in range(n*q):
                        tmp = torch.matmul(v.t(), x_after[:,k])/(torch.norm(x_after[:,k], p=2)*torch.norm(v, p=2)+1e-3)
                        Jv.append(tmp.item())
                    if max(Jv) < epsilon:
                        self.chosen_index.append((count,)+index)
                        out.append(v)
                        count += 1
                if count == m:
                    break
        else:
            out = []

            # test mode
            print("-----------Attention: test mode-----------")
            self.chosen_index = [self.f.iloc[row][1:].to_numpy() for row in range(self.f.shape[0])]
            self.chosen_index = np.array(self.chosen_index)

            for index in self.chosen_index:
                D1 = x[:, index[1], index[2]]
                D2 = x[:, index[3], index[4]]
                D = torch.stack([D1, D2], 1)
                v = torch.min(D,1)[0]
                out.append(v)
        min_out = torch.stack(out,1)
        return min_out

    def forward(self, x, init_struct):
        '''
        x: (batch_size, input_size=num_nodes*num_features)
        max_x: (batch_size, input_size, quantile)
        min_x: (batch_size, intemediate_nodes)
        output: (batch_size, output_size)
        '''
        max_x = self.Maxout_Layer(x, 1)
        max_x = max_x.permute(1,2,0)
        min_x = self.Minout_Layer(max_x, init_struct)
        x = torch.cat([torch.flatten(max_x, start_dim=1), min_x], 1)
        output = torch.add(torch.matmul(x, self.w), self.biases)
        return output, self.w, max_x, min_x, self.chosen_index

class EHH_Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, intermediate_node=20, quantile=6, normalization=True, file_path='models/adj30.csv'):
        super(EHH_Net, self).__init__()
        self.normalization = normalization
        self.hidden_size = hidden_size
        self.embedding = Encoder(input_size, hidden_size)
        self.EHH_layer = EHH_Layer(hidden_size, output_size, intermediate_node, quantile, file_path)
        self.adj = self.EHH_layer.chosen_index

    def normalization_layer(self, data):
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        normed = (data-data_min)/(data_max-data_min+1e-5)
        return normed

    def ANOVA(self, w, max_x, min_x):
        '''
        max_neuron = ehh_input_size*quantile
        max_x: (batch_size, ehh_input_size, quantile)
        min_x: (batch_size, min_neurons)
        w: (max_neurons + min_neurons, output_size)
        sigma: (max_x.shape[1]+num_min_neurons, out_size)
        '''
        num_max_neurons = max_x.shape[1]*max_x.shape[2]
        num_min_neurons = min_x.shape[1]
        q = max_x.shape[2]
        sigma = torch.zeros([max_x.shape[1]+num_min_neurons, w.shape[1]])

        for i in range(max_x.shape[1]):
            '''
            zi: (batch_size, quantile)
            wi: (quantile, out_size)
            '''
            zi = max_x[:,i,:]
            wi = w[i*q:(i+1)*q,:]
            fi = torch.matmul(zi, wi)
            sigma[i] = torch.sqrt(torch.var(fi, dim=0))
        
        for j in range(num_min_neurons):
            zj = min_x[:,j][:, None]
            wj = w[num_max_neurons+j,:][None,:]
            fj = torch.matmul(zj, wj)
            sigma[max_x.shape[1]+j] = torch.sqrt(torch.var(fj, dim=0))

        return sigma, self.linear_w, self.linear_b

    def forward(self, x, init_struct):
        if self.normalization:
            normed = self.normalization_layer(x)
        else:
            normed = x
        embedding_x, self.linear_w, self.linear_b = self.embedding(normed)
        output, w, max_x, min_x, adj = self.EHH_layer(embedding_x, init_struct)
        return embedding_x, output, w, max_x, min_x, adj