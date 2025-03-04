'''
@ Summary: Deep Graph Convolutional Reinforcement Learning Model
@ Author: XTBJ
'''
import math,random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
# Indefinite length parameter
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def add(self, obs, action, reward, new_obs, matrix, new_matrix, done):
        experience = (obs, action, reward, new_obs, matrix, new_matrix, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

class Encoder(nn.Module):
    def __init__(self, dim=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding

class AttModel(nn.Module):
    def __init__(self, n_node, dim, hidden_dim, dout):
        super(AttModel, self).__init__()
        self.fcv = nn.Linear(dim, hidden_dim)
        self.fck = nn.Linear(dim, hidden_dim)
        self.fcq = nn.Linear(dim, hidden_dim)
        self.fout = nn.Linear(hidden_dim, dout)

    def forward(self, x, mask):
        v = F.relu(self.fcv(x))
        q = F.relu(self.fcq(x))
        k = F.relu(self.fck(x)).permute(0,2,1)
        att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1-mask), dim=2)

        out = torch.bmm(att, v)
        out = F.relu(self.fout(out))
        return out

class Q_Net(nn.Module):
    def __init__(self, hidden_dim, dout):
        super(Q_Net, self).__init__()
        self.fc = nn.Linear(hidden_dim, dout)

    def forward(self, x):
        q = self.fc(x)
        return q

class DGN(nn.Module):
    def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
        super(DGN, self).__init__()
        self.encoder = Encoder(num_inputs, hidden_dim)
        # multi-agent
        self.att1 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.att2 = AttModel(n_agent, hidden_dim, hidden_dim, hidden_dim)
        self.q_net = Q_Net(hidden_dim, num_actions)

    def forward(self, x, mask):
        '''
        x: (batch_size, n_agents, input_size)
        h1: (batch_size, n_agents, hidden_size)
        '''
        h1 = self.encoder(x)
        h2 = self.att1(h1, mask)
        h3 = self.att2(h2, mask)
        q = self.q_net(h3)
        return q