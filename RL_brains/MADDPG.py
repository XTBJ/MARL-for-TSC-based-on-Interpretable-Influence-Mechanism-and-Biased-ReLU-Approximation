'''
@ Summary: MADDPG + MLP + ReLU
@ Author: XTBJ
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

torch.autograd.set_detect_anomaly(True)

def one_hot(label, depth=4):
    # print("size:", label.size(0)) 
    out = torch.zeros(label.size(0), depth)
    for k in range(label.size(0)):
        out[k, int(label[k, :])] = 1
    return out

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, n_state, n_action, n_hidden):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(n_state, n_hidden)
        self.linear_a2 = nn.Linear(n_hidden, n_action)
        self.reset_parameters()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tahn = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain_tahn)
    def forward(self, input):
        x = self.LReLU(self.linear_a1(input))
        policy = self.tanh(self.linear_a2(x))
        return policy

class critic_agent(abstract_agent):
    def __init__(self, n_state, n_action, n_hidden):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(n_state, n_hidden)
        self.linear_a_c1 = nn.Linear(n_action, n_hidden)
        self.linear_c2 = nn.Linear(n_hidden*2, n_hidden)
        self.linear_c = nn.Linear(n_hidden, 1)
        self.reset_parameters()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tahn = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)
    def forward(self, obs_input, action_input):
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class MADDPG_Net(nn.Module):
    def __init__(self, ts_ids, n_state, n_hidden, n_action, lr_a, lr_c, gamma, tao=0.01):
        super(MADDPG_Net, self).__init__()
        self.n_agent = len(ts_ids)
        self.actor_eval = [actor_agent(n_state, n_action, n_hidden) for _ in range(self.n_agent)]
        self.actor_tar = [actor_agent(n_state, n_action, n_hidden) for _ in range(self.n_agent)]
        self.critic_eval = [critic_agent(n_state*self.n_agent, n_action*self.n_agent, n_hidden) for _ in range(self.n_agent)]
        self.critic_tar = [critic_agent(n_state*self.n_agent, n_action*self.n_agent, n_hidden) for _ in range(self.n_agent)]
        self.optimizer_a = [torch.optim.Adam(self.actor_eval[i].parameters(), lr=lr_a) for i in range(self.n_agent)]
        self.optimizer_c = [torch.optim.Adam(self.critic_eval[i].parameters(), lr=lr_c) for i in range(self.n_agent)]
        self.actor_tar = self.update_trainer(self.actor_eval, self.actor_tar, 1.0)
        self.critic_tar = self.update_trainer(self.critic_eval, self.critic_tar, 1.0)
        self.gamma = gamma
        self.tao = tao
        self.learn_step_counter = 0
    
    def update_trainer(self, agent_eval, agent_tar, tao):
        for agent_e, agent_t in zip(agent_eval, agent_tar):
            key_list = list(agent_e.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_e = agent_e.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_e[key]*tao + (1-tao)*state_dict_t[key]
            agent_t.load_state_dict(state_dict_t)
        return agent_tar

    def learn(self, sample):
        '''
            Update centralized critic for all agents
            obs: (num_agents, batch_size, obs_dim)
            acs: (num_agents, batch_size, acs_dim)
        '''
        obs, acs, rews, next_obs, dones, waiting_time = sample

        acs_dim = acs[0].shape[1]*4
        batch_size = acs[0].shape[0]

        obs_in = torch.cat(obs, dim=0).reshape(self.n_agent, batch_size, -1)
        obs_in = obs_in.permute(1,0,2).reshape(batch_size, -1)
        obs_next_in = torch.cat(next_obs, dim=0).reshape(self.n_agent, batch_size, -1)
        obs_next_in = obs_next_in.permute(1,0,2).reshape(batch_size, -1)
        ac_in = torch.cat(acs, dim=0).reshape(self.n_agent, batch_size, -1)
        ac_in =  ac_in.permute(1,0,2).reshape(batch_size, -1)

        for ind, actor_e, actor_t, critic_e, critic_t, opt_a, opt_c in zip(range(self.n_agent), self.actor_eval, self.actor_tar, self.critic_eval, self.critic_tar, self.optimizer_a, self.optimizer_c):
            # update critic
            q = critic_e(obs_in, torch.cat([one_hot(a) for a in acs], dim=1))
            with torch.no_grad():
                q_ = critic_t(obs_next_in, torch.cat([a_t(next_obs[i]) for i, a_t in enumerate(self.actor_tar)], dim=1))
                target_value = q_*self.gamma + rews[ind]
            loss_c = torch.nn.MSELoss()(q, target_value)
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

            # update actor
            policy_c_new = actor_e(obs[ind])
            ac_one_hot = torch.cat([one_hot(a) for a in acs], dim=1)
            ac_one_hot[:, acs_dim*ind:acs_dim*(ind+1)] = policy_c_new
            loss_a = torch.mul(-1, torch.mean(critic_e(obs_in, ac_one_hot)))

            opt_a.zero_grad()
            loss_a.backward()
            opt_a.step()
            


