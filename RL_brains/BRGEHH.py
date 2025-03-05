'''
@ Summary: EHH + Graph + PPO +  ANOVA Decomposition + retrained EHH neural network.
Author: XTBJ
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from NN.BReLU import BReLU_Net
from NN.EHH import EHH_Net
from parameters import *

torch.autograd.set_detect_anomaly(True)
MSELoss = torch.nn.MSELoss()

INTERMEDIATE_NODES = 150

def one_hot(label, depth=4):
    # print("size:", label.size(0))
    out = torch.zeros(label.size(0), depth)
    for k in range(label.size(0)):
        out[k, int(label[k, :])] = 1
    out = out.cuda()
    return out

def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs

# Actor
class PolicyNet(nn.Module):
    def __init__(self, n_state, n_hidden, n_action, q, mean=0, var=1):
        super(PolicyNet, self).__init__()
        self.BReLU_Layer = BReLU_Net(n_state, 10, 10, n_action, q, mean, var)

    def forward(self, x):
        out, w1, w2, w3 = self.BReLU_Layer(x)
        out = F.softmax(out, dim=1)
        return out, w1, w2, w3

class BReLU_Critic(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, q, mean=0, var=1):
        super(BReLU_Critic, self).__init__()
        self.BReLU_Layer = BReLU_Net(n_in, 12, 12, n_out, q, mean, var)
    def forward(self, x):
        out, w1, w2, w3 = self.BReLU_Layer(x)
        return out, w1, w2, w3

class ValueNet(nn.Module):
    def __init__(self, n_sa, n_hidden, checkpoint, q, EHH_adj_file, mean=0, var=1, norm_in=True, episode=10, intermediate_node=INTERMEDIATE_NODES, quantile=6):
        super(ValueNet, self).__init__()
        '''
        Input: state + action
               n_sa (list of (int,int))
        # The influence of each agent's feature on each intersection
        self.ehh(input_size=num_agents*n_s, hidden_size=ehh_hidden_size, out_size=num_agents)
        '''
        self.n_sa = n_sa
        
        self.state_encoders = nn.ModuleList()
        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.episode = episode
        self.n_hidden = n_hidden
        self.source_node = n_hidden*quantile*n_sa[0][1]
        self.intermediate_node = intermediate_node
        
        n_input = 0
        i = 0
        for n_s, n_a in n_sa:
            in_dim = n_s + n_a
            out_dim = n_a
            n_input += n_s

            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(in_dim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(in_dim, n_hidden))
            encoder.add_module('enc_n1', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            
            self.critics.append(BReLU_Critic(len(n_sa)*n_hidden, n_hidden, out_dim, q=q, mean=mean[i*n_hidden:(i+1)*n_hidden], var=var[i*n_hidden:(i+1)*n_hidden]))

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(n_s, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(n_s, n_hidden))
            state_encoder.add_module('s_enc_n1', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)
            i += 1

        self.n_agent = len(self.critic_encoders)
        self.ehh_hidden = n_hidden*self.n_agent

        removed_keys = ['linear_w', 'linear_b']
        for key in removed_keys:
            if key in checkpoint:
                checkpoint.pop(key)
        
        self.ehh = EHH_Net(input_size=n_input, hidden_size=self.ehh_hidden, output_size=self.n_agent, intermediate_node=self.intermediate_node, normalization=True, file_path=EHH_adj_file)
        self.ehh.load_state_dict(checkpoint)
        
    def forward(self, inps, init_struct_update=False, agents=None, return_q=True, return_all_q=False, regularize=False, return_attend=False):
        '''
        Input: 
            inps: batch of obseravtion + action
            next_inps: batch of next observation + action
            agents (int): indices of agents to return Q for
            states: (num_nodes, batch_size, num_features)
            ehh_s: (batch_size, num_nodes*num_features)
            anova: (num_nodes, ehh_hidden+intermediate_nodes)
            s_encoding: (num_nodes, batch_size, hidden_size)
            sa_encoding: (num_nodes, batch_size, hidden_size)
        '''

        if agents is None:
            agents = range(len(self.critic_encoders))

        # [n, batch size, feature size]
        states = [s for s,a in inps]
        actions = [a for s,a in inps]
        
        inps = [torch.cat((s, one_hot(a, depth=n_a)), dim=1) for (s,a),(n_s,n_a) in zip(inps, self.n_sa)]

        # extract state-action encoding for each agent 
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # obs + acs --> after encoding
        s_vector = torch.flatten(torch.stack(s_encodings, dim=1), start_dim=1)
        sa_vector = torch.flatten(torch.stack(sa_encodings, dim=1), start_dim=1)

        ehh_s = torch.cat(states, dim=1)
        batch_size = sa_vector.size()[0]     
        ehh_embedding, pred, w, max_x, min_x, adj = self.ehh(ehh_s, init_struct=False)

        anova, linear_w, linear_b = self.ehh.ANOVA(w, max_x, min_x)
        anova = torch.tensor(anova, dtype=torch.float32)

        # ----------- version-2 (EHH_v9) -----------
        '''
        out_size = n_agents
        all_att: (out_size, ehh_hidden)
        att_s: (batch_size, out_size, ehh_hidden)
        att_out: (batch_size, out_size, ehh_hidden)
        '''
        self_att = anova[0:self.ehh_hidden, :]
        bi_att = anova[self.ehh_hidden:, :]
        neighbor_att = torch.zeros([self.ehh_hidden, self.n_agent], dtype=torch.float32)
        for ind in adj:
            val = bi_att[ind[0], :]
            neighbor_att[ind[1], :] = val
            neighbor_att[ind[3], :] = val
        all_att = (self_att + neighbor_att).t().cuda()
        att_s = ehh_embedding
        att_s = att_s.repeat(self.n_agent, 1, 1).permute(1,0,2)
        att_s = att_s.reshape(batch_size, self.n_agent, -1)
        att_out = att_s*all_att
        all_value = att_out.permute(1,0,2)
        
        # calculate Q per agents
        all_rets = []
        for i, a_i in enumerate(agents):
            critic_in = all_value[i]
            all_q, w1, w2, w3 = self.critics[a_i](critic_in)
            in_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, in_acs)
            if return_q:
                all_rets.append(q)
            if return_all_q:
                all_rets.append(all_q)
        return all_rets, w1, w2, w3

class BRGEHHNet(nn.Module):
    def __init__(self, n_agent, n_state, n_hidden, n_action, lr_a, lr_c, gamma, lmbda, epochs, eps, device, EHH_model_file, EHH_adj_file, q=1):
        super(BRGEHHNet, self).__init__()
        checkpoint = torch.load(EHH_model_file)
        # mean and var in BReLU are obtained by the Offline EHH_layer
        self.mean = checkpoint['EHH_layer.batch_layer.gamma']
        self.var = checkpoint['EHH_layer.batch_layer.beta']
        self.gamma = gamma
        self.actors = [PolicyNet(n_s, n_hidden, n_a, q=q, mean=self.mean[i*n_hidden:(i+1)*n_hidden], var=self.var[i*n_hidden:(i+1)*n_hidden]).to(device) for i, n_s, n_a in zip(range(len(n_agent)), n_state, n_action)]
        self.actor_optimizers = [torch.optim.Adam(self.actors[i].parameters(), lr=lr_a) for i in range(len(n_agent))]
        self.old_actors = [PolicyNet(n_s, n_hidden, n_a, q=q, mean=self.mean[i*n_hidden:(i+1)*n_hidden], var=self.var[i*n_hidden:(i+1)*n_hidden]).to(device) for i, n_s, n_a in zip(range(len(n_agent)),n_state, n_action)]
        n_sa = [(n_s, n_a) for n_s, n_a in zip(n_state, n_action)]
        self.n_action = n_action
        self.critic = ValueNet(n_sa, n_hidden, q=q, EHH_adj_file=EHH_adj_file, mean=self.mean, var=self.var, checkpoint=checkpoint)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.n_agent = n_agent # ts_id
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # GAE scale factor
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.learn_step_counter = 0
        self.aloss = []

    def hard_update(self, target, source):
        '''
        Copy network parameters from source to target
        '''
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data*(1.0-TAU) + param.data*TAU)

    def choose_action(self, state, EPSILON=0.9):
        # action --> dictionary {'ts1': 0-3 ...}
        action = {}
        for i in range(len(self.n_agent)):
            # numpy[n_state] --> numpy[1,n_state] --> tensor
            if np.random.uniform() < EPSILON:   # optimal action
                s = state[self.n_agent[i]]
                s = torch.tensor(s[np.newaxis, :], dtype=torch.float).to(self.device)
                probs, w1, w2, w3 = self.actors[i](s)
                action_dist = torch.distributions.Categorical(probs)
                action[self.n_agent[i]] = action_dist.sample().item()
            else:   # random action
                action[self.n_agent[i]] = np.random.randint(0, self.n_action[i])
        return action

    def prep_training(self):
        self.critic.train()
        for i in range(len(self.n_agent)):
            self.actors[i].train()


    def learn(self, sample, init_struct_update=False):
        '''
            Update centralized critic for all agents
        '''
        obs, acs, rews, next_obs, dones, waiting_time = sample

        # update the old pi
        if self.learn_step_counter % REPLACE_ITER == 0:
            # simple copy
            for old_actor, actor in zip(self.old_actors, self.actors):
                old_actor.load_state_dict(actor.state_dict())

        self.learn_step_counter += 1
        
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.old_actors, next_obs):
            probs = pi(ob)[0]
            curr_next_ac, curr_next_log_pi = categorical_sample(probs, use_cuda=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        next_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))

        # EHH
        next_q_target, next_cw1, next_cw2, next_cw3 = self.critic(next_critic_in)
        td_value, cw1, cw2, cw3 = self.critic(critic_in)

        td_delta = []
        adv_all = []
        old_probs = []
        for a_i, next_q, log_pi, q in zip(range(len(self.n_agent)), next_q_target, next_log_pis, td_value):
            td_target = (rews[a_i].view(-1,1) + self.gamma*next_q*(1-dones[a_i].view(-1,1)))
            # compute the advantage function
            # inverse sequence differential value
            td_delta.append(td_target - q)
            td_delta[a_i] = td_delta[a_i].cpu().detach().numpy()
            advantage = 0
            advantage_list = []
            for delta in td_delta[::-1]:
                advantage = self.gamma*self.lmbda*advantage + delta
                advantage_list.append(advantage)
            # positive sequence
            advantage_list.reverse()
            # numpy --> tensor [batch_size, 1]
            advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
            adv_all.append(advantage)

            # Obtain the probability of the current action
            acs_tmp = acs[a_i].type(torch.int64)
            old_probs_tmp = self.old_actors[a_i](obs[a_i])[0].gather(1, acs_tmp.unsqueeze(-1).squeeze(-1)).detach()
            old_probs.append(old_probs_tmp)
        
        probs = []
        ratio = []
        surr1 = []
        surr2 = []
        actor_loss = []
        entropy = []
        closs = []
        critic_loss = 0

        for a_i, next_q, log_pi, q in zip(range(len(self.n_agent)), next_q_target, next_log_pis, td_value):
            td_target = (rews[a_i].view(-1,1) + self.gamma*next_q*(1-dones[a_i].view(-1,1)))
            closs.append(MSELoss(q, td_target.detach()))
            
            acs_tmp = acs[a_i].type(torch.int64)
            pred, w1, w2, w3 = self.actors[a_i](obs[a_i])
            probs_tmp = pred.gather(1, acs_tmp.unsqueeze(-1).squeeze(-1))
            probs.append(probs_tmp)
            ratio.append(probs[a_i]/(old_probs[a_i]+1e-5))
            surr1.append(ratio[a_i]*adv_all[a_i])
            surr2.append(torch.clamp(ratio[a_i], 1-self.eps, 1+self.eps)*adv_all[a_i])
            loss_tmp = torch.mean(-torch.min(surr1[a_i], surr2[a_i])) + 0.05*torch.sum(torch.abs(w1))+ 0.03*torch.sum(torch.abs(w2))+0.01*torch.sum(torch.abs(w3))
            
            actor_loss.append(loss_tmp)
            self.actor_optimizers[a_i].zero_grad()
            actor_loss[a_i].backward(retain_graph=True)
            self.actor_optimizers[a_i].step()
        
        for loss in closs:
            critic_loss += loss
        critic_loss += 0.05*torch.sum(torch.abs(cw1))+0.03*torch.sum(torch.abs(cw2))+0.01*torch.sum(torch.abs(cw3))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        