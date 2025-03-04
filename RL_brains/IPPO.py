'''
@ Summary: Independent PPO + MLP + ReLU
@ Author: XTBJ
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, dim=32, hidden_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        embedding = F.relu(self.fc(x))
        return embedding

# class EHH_AttModule(nn.Module):
#     def __init__(self, )

# Actor
class PolicyNet(nn.Module):
    def __init__(self, n_state, n_hidden, n_action):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_action)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out

# Critic
class ValueNet(nn.Module):
    def __init__(self, n_state, n_hidden):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)
        return out

# PPO
class PPO_Net(nn.Module):
    def __init__(self, n_state, n_hidden, n_action, lr_a, lr_c, gamma, lmbda, epochs, eps, device):
        super(PPONet, self).__init__()
        self.gamma = gamma
        # one agent
        self.actor = PolicyNet(n_state, n_hidden, n_action).to(device)
        self.critic = ValueNet(n_state, n_hidden).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # GAE scale factor
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def choose_action(self, state):
        # numpy[n_state] --> numpy[1,n_state] --> tensor
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def learn(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).to(self.device).view(-1,1)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).to(self.device).view(-1,1)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).to(self.device).view(-1,1)
        # [batch_size, 1]
        next_q_target = self.critic(next_state)
        td_target = reward + self.gamma*next_q_target*(1-done)
        td_value = self.critic(state)
        td_delta = td_target - td_value

        # tensor --> numpy
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        # compute the advantage function
        # inverse sequence differential value
        for delta in td_delta[::-1]:
            advantage = self.gamma*self.lmbda*advantage + delta
            advantage_list.append(advantage)
        # positive sequence
        advantage_list.reverse()
        # numpy --> tensor [batch_size, 1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # Obtain the probability of the current action
        old_log_probs = torch.log(self.actor(state).gather(1, action)).detach()
        old_log_probs = torch.clamp(old_log_probs, min=1e-7)

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(state).gather(1, action))
            log_probs = torch.clamp(log_probs, min=1e-7)
            # entropy = -torch.sum(self.actor(state)*torch.log(self.actor(state) + 1e-8), dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps)*advantage
            
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))
            # print("actor_loss:", actor_loss)
            # print("critic_loss:", critic_loss)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # return actor_loss, critic_loss, entropy
