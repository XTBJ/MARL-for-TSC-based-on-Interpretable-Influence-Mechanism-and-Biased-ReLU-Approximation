'''
@ Summary: Independent Deep Q Network + MLP + ReLU
@ Author: XTBJ
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, n_state, n_action, n_hidden):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, n_hidden, bias=True),
            nn.Linear(n_hidden, n_action, bias=True),
            nn.ReLU()
        )
    def forward(self, s):
        q = self.net(s)
        return q        


class DQN(nn.Module):
    def __init__(self, n_state, n_hidden, n_action, lr, reward_decay, replace_target_iter, memory_size, batch_size, e_greedy):
        super(DQN, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.n_hidden = n_hidden
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size

        # total learning step
        self.learn_step_counter = 0

        # initialize zeo memory [s,a,r,s_]
        self.memory = np.zeros((self.memory_size, self.n_state*2 + 2))

        self.eval_net = Network(self.n_state, self.n_action, self.n_hidden)
        self.target_net = Network(self.n_state, self.n_action, self.n_hidden)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        
        # record the error of each step
        self.cost_his = []

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s,[a,r],s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            actions_value = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            action = np.random.randint(0, self.n_action)
        return action
    
    def _replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('target params replaced')
            
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        s = torch.FloatTensor(batch_memory[:, :self.n_state])
        s_ = torch.FloatTensor(batch_memory[:, -self.n_state:])
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_state].astype(int)
        reward = batch_memory[:, self.n_state + 1]

        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values

        # train the eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().numpy())

        self.learn_step_counter += 1

