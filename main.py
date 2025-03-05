import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import argparse
import pandas as pd
import numpy as np
from gym import spaces
from datetime import datetime
from environment.SUMO_env import SumoEnvironment
from environment.buffer import ReplayBuffer
from RL_brains.DGN import ReplayBuffer as DGN_BUFFER
from parameters import *

from RL_brains.IDQN import DQN
from RL_brains.IPPO import PPO_Net
from RL_brains.MADDPG import MADDPG_Net
from RL_brains.DGN import DGN
from RL_brains.BRGEHH import BReLU_Net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_name', type=str, help='name of the TSC method: 1. IDQN; 2. IPPO; 3. MADDPG; 4. DGN; 5. BRGEHH')
    parser.add_argument('--env_name', type=str, default='noneuclidean', help='name of the simulation environment: 1. grid; 2. noneuclidean; 3. cologne')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training')
    parser.add_argument('--use_gui', type=boolean_string, default=False, help='open the sumo graphical interface')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'

def runner():
    args = parse_args()
    method_name = args.method_name
    env_name = args.env_name
    train = args.train
    use_gui = args.use_gui
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # =========================================================
    #                 CREATING THE SIMULATION
    # =========================================================

    if env_name == 'grid':
        net_file = "networks/Grid/grid5x5.net.xml"
        route_file = "networks/Grid/routes.rou.xml"
        EHH_adj_file = "models/Grid.csv"
        EHH_model_file = "models/Grid.pth"
    elif env_name == 'noneuclidean':
        net_file = "networks/NonEuclidean/NonEuclidean.net.xml"
        route_file = "networks/NonEuclidean/routes.rou.xml"
        EHH_adj_file = "models/NonEuclidean.csv"
        EHH_model_file = "models/NonEuclidean.csv"
    elif env_name == 'cologne':
        net_file = "networks/Cologne/cologne.net.xml"
        route_file = "networks/Cologne/cologne6to8.trips.xml"
        EHH_adj_file = "models/Cologne.csv"
        EHH_model_file = "models/Cologne.pth"

    env = SumoEnvironment(net_file=net_file, route_file=route_file, use_gui=use_gui, min_green=MIN_GREEN_TIME, delta_time=DELTA_TIME, num_seconds=NUM_SECONDS)
    replay_buffer = ReplayBuffer(max_steps=MAX_STEPS, num_agents=len(env.ts_ids), obs_dims=[env.observation_spaces(ts).shape[0] for ts in env.ts_ids], ac_dims=[0 for ts in env.ts_ids])

    # =========================================================
    #                       ALGORITHM
    # =========================================================

    if method_name == "IDQN":
        agent = {ts: DQN(n_state=env.observation_spaces(ts).shape[0], n_hidden=HIDDEN_SIZE, n_action=env.action_spaces(ts).n, lr=Q_LR, reward_decay=REWARD_DECAY, replace_target_iter=REPLACE_TARGET_ITER, memory_size=MEMORY_SIZE, batch_size=batch_size, e_greedy=EPSILON) for ts in env.ts_ids}
    elif method_name == "IPPO":
        agent = {ts: PPO_Net(n_state=env.observation_spaces(ts).shape[0], n_hidden=HIDDEN_SIZE, n_action=env.action_spaces(ts).n, lr_a=ACTOR_LR, lr_c=CRITIC_LR, gamma=GAMMA, lmbda=LMBDA, epochs=EPOCHS, eps=POLICY_CLIP, device=device) for ts in env.ts_ids}
    elif method_name == "MADDPG":
        agent = MADDPG_Net(ts_ids=env.ts_ids, n_state=[env.observation_spaces(ts).shape[0] for ts in env.ts_ids], n_hidden=HIDDEN_SIZE, n_action=[env.action_spaces(ts).n for ts in env.ts_ids], lr_a=ACTOR_LR, lr_c=CRITIC_LR, gamma=GAMMA)
    elif method_name == "DGN":
        agent = DGN(n_agent=len(env.ts_ids), num_inputs=[env.observation_spaces(ts).shape[0] for ts in env.ts_ids], hidden_dim=HIDDEN_SIZE, num_actions=[env.action_spaces(ts).n for ts in env.ts_ids])
    elif method_name == "BRGEHH":
        agent = BRGEHHNet(n_agent=env.ts_ids, n_state=[env.observation_spaces(ts).shape[0] for ts in env.ts_ids], n_hidden=HIDDEN_SIZE, n_action=[env.action_spaces(ts).n for ts in env.ts_ids], lr_a=ACTOR_LR, lr_c=CRITIC_LR, gamma=GAMMA, lmbda=LMBDA, epochs=EPOCHS, eps=POLICY_CLIP, device=device, EHH_model_file=EHH_model_file, EHH_adj_file=EHH_adj_file, q=QUARTILE)
    else:
        sys.exit()
    
    # =========================================================
    #                        TRAINING
    # =========================================================
    
    if method_name == "DGN":
        if train:
            buff = DGN_BUFFER(buffer_size)(CAPACITY)
            n_agent = len(env.ts_ids)
            model = DGN(n_agent, [env.observation_spaces(ts).shape[0] for ts in env.ts_ids], HIDDEN_SIZE, [env.action_spaces(ts).n for ts in env.ts_ids]).to(device)
            model_tar = DGN(n_agent, [env.observation_spaces(ts).shape[0] for ts in env.ts_ids], HIDDEN_SIZE, [env.action_spaces(ts).n for ts in env.ts_ids]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=Q_LR)
            Matrix = np.ones((BATCH_SIZE, n_agent, n_agent))
            Next_Matrix = np.ones((BATCH_SIZE, n_agent, n_agent))

            for ep in range(EPISODE_LENGTH):
                step = 0
                score = 0
                obs = env.reset()

                state = []
                for junction in env.ts_ids:
                    state.append(torch.tensor(obs[junction], dtype=torch.float).unsqueeze(0))
                '''
                obs: (n_agent, n_state) [list (tensor)]
                '''
                obs = state
                infos = []
                new_infos = {key: [] for key in env.ts_ids}
                while step < MAX_STEP:
                    step += 1
                    action = {}
                    adj = env.adj

                    q = model(obs, torch.Tensor(np.array([adj])))
                    for i in range(n_agent):
                        if np.random.rand() < epsilon:
                            a = np.random.randint(n_actions[i])
                        else:
                            a = q[i].argmax().item()
                        action[env.ts_ids[i]] = a

                    next_obs, reward, done, info = env.step(action)
                    next_adj = env.adj

                    next_state = []
                    for junction in env.ts_ids:
                        next_state.append(torch.tensor(next_obs[junction], dtype=torch.float).unsqueeze(0))
                    next_obs = next_state

                    waiting_time = env.compute_waiting_time()

                    buff.add(obs, action, reward, next_obs, adj, next_adj, done)
                    obs = next_obs
                    adj = next_adj
                    for junction in env.ts_ids:
                        score += reward[junction]

                    sum_reward = 0
                    for ts in env.ts_ids:
                        sum_reward += reward[ts]

                    info = {'step_time':step, 'waiting_time': waiting_time, 'reward': sum_reward}
                    infos.append(info)

                    # new infos record
                    current_info = env._compute_step_info()
                    for ts in env.ts_ids:
                        current_dict = current_info[ts]
                        current_dict["reward"] = reward[ts]
                        current_dict["action"] = action[ts]
                        new_infos[ts].append(current_dict)

                    print("step:", step, sum_reward)

                for n in range(train_ep):
                    '''
                    batch: (samples) -- obs, next_obs, matrix, next_matrix)
                    O: [n_agent, (batch, o_dim)]
                    '''
                    batch = buff.getBatch(batch_size)

                    O = [[] for _ in range(n_agent)]
                    Next_O = [[] for _ in range(n_agent)]
                    for j in range(batch_size):
                        sample = batch[j]
                        Matrix[j] = sample[4]
                        Next_Matrix[j] = sample[5]
                        for i in range(n_agent):
                            O[i].append(sample[0][i])
                            Next_O[i].append(sample[3][i])

                    for i in range(n_agent):
                        O_tmp = O[i]
                        O_tmp = torch.stack(O_tmp, dim=0)
                        O_tmp = torch.squeeze(O_tmp)
                        O[i] = O_tmp

                        Next_O_tmp = Next_O[i]
                        Next_O_tmp = torch.stack(Next_O_tmp, dim=0)
                        Next_O_tmp = torch.squeeze(Next_O_tmp)
                        Next_O[i] = Next_O_tmp
                    
                    q_value = model(O, torch.Tensor(Matrix).to(device))
                    target_q = model_tar(Next_O, torch.Tensor(Next_Matrix).to(device))

                    target_q_value = []
                    expected_q = []
                    for i in range(n_agent):
                        target_q_value.append(target_q[i].detach().cpu().numpy())
                        expected_q.append(q_value[i].detach().cpu().numpy().copy())

                    index = []
                    for j in range(batch_size):
                        sample = batch[j]
                        for i in range(n_agent):
                            if sample[6][env.ts_ids[i]]:
                                flag = 1
                            else:
                                flag = 0
                            tmp = sample[2][env.ts_ids[i]] + (1-flag)*GAMMA*np.max(target_q_value[i][j])
                            expected_q[i][j,sample[1][env.ts_ids[i]]] = tmp

                    loss_tmp = [(q-torch.tensor(eq, dtype=torch.float)).pow(2).mean() for q,eq in zip(q_value, expected_q)]
                    loss = sum(loss_tmp)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                for ts in env.ts_ids:
                    df = pd.DataFrame(new_infos[ts])
                    df.to_csv('results/' + env_name + '/DGN_' + str(ep) + '_' + ts + '.csv')

            # save the model
            torch.save(model.state_dict(),"checkpoints/DGN.pt")

    else:       
        if train:
            for ep in range(EPISODE_LENGTH):
                inital_state = env.reset()
                s = initial_states
                step = 0
                infos = {key: [] for key in env.ts_ids}
                transition_dict = {ts:{
                    'state':[],
                    'action':[],
                    'reward':[],
                    'next_state':[],
                    'done':[],
                    'waiting_time':[],
                } for ts in env.ts_ids}
                
                while step < TIME_STEP:
                    step += 1

                    if method_name == "IDQN" or method_name == "IPPO":
                        a = {ts: agent[ts].choose_action(s[ts]) for ts in env.ts_ids}
                    elif method_name == "MADDPG":
                        a = {ts: np.argmax(agent(torch.from_numpy(s[ts]).to(torch.float)).detach().cpu().numpy()) for agent, ts in zip(agent.actor_eval, env.ts_ids)}
                    elif method_name == "BRGEHH":
                        a = agent.choose_action(s)


                    s_, r, done, feedback = env.step(action=a)
                    waiting_time = env.compute_waiting_time()

                    for ts in env.ts_ids:
                        transition_dict[ts]['state'].append(s[ts])
                        transition_dict[ts]['action'].append(a[ts])
                        transition_dict[ts]['reward'].append(r[ts])
                        transition_dict[ts]['next_state'].append(s_[ts])
                        transition_dict[ts]['done'].append(done[ts])
                        transition_dict[ts]['waiting_time'].append(waiting_time[ts])

                    state_tmp = []
                    action_tmp = []
                    reward_tmp = []
                    next_state_tmp = []
                    done_tmp = []
                    waiting_time_tmp = []
                    for ts in env.ts_ids:
                        state_tmp.append(s[ts])
                        action_tmp.append([a[ts]])
                        reward_tmp.append([r[ts]])
                        next_state_tmp.append(s_[ts])
                        done_tmp.append([done[ts]])
                        waiting_time_tmp.append([waiting_time[ts]])

                    replay_buffer.push(np.array(state_tmp)[None,:], np.array(action_tmp)[None,:], np.array(reward_tmp)[None,:], np.array(next_state_tmp)[None,:], np.array(done_tmp)[None,:], np.array(waiting_time_tmp)[None,:])

                    if method_name == "IDQN":
                        for ts in env.ts_ids:
                            agent[ts].store_transition(s[ts], a[ts], r[ts], s_[ts])
                        if (step>100) and (step%5==0):
                            for ts in env.ts_ids:
                                agent[ts].learn()
                    elif method_name == "IPPO":
                        for ts in env.ts_ids:
                            agent[ts].learn(transition_dict[ts])
                    elif method_name == "MADDPG" or method_name == "BRGEHH":
                        if (len(replay_buffer) >= BATCH_SIZE):
                            sample = replay_buffer.sample(BATCH_SIZE, to_gpu=True)
                            agent.learn(sample)
                    else:
                        sys.exit()
                
                    reward = 0.0
                    all_waiting_time = 0.0
                    for ts in env.ts_ids:
                        reward += r[ts]
                        all_waiting_time += waiting_time[ts]
                    
                    current_info = env._compute_step_info()
                    for ts in env.ts_ids:
                        current_dict = current_info[ts]
                        current_dict["reward"] = r[ts]
                        current_dict["action"] = a[ts]
                        infos[ts].append(current_dict)

                    s = s_
                    print("step:%d, reward:%d, waiting time:%d" %(step, reward, all_waiting_time))

                    for ts in env.ts_ids:
                        df = pd.DataFrame(infos[ts])
                        df.to_csv('results/' + env_name + "/" + method_name + '_' + str(ep) + '_' + ts + '.csv')
            
            # save the model
            torch.save(agent.state_dict(), "checkpoints/" + method_name + ".pt")

if __name__ == '__main__':
    runner()

