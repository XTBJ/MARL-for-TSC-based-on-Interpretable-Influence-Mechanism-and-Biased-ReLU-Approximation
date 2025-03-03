'''
sumo simulation
return state action reward
'''
import os
import sys

from matplotlib.pyplot import axis
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import time
import traci
import sumolib
import numpy as np
import pandas as pd
from traffic_signal import TrafficSignal
from traci.main import simulationStep

class SumoEnvironment():
    def __init__(self, net_file, route_file, out_csv_name=None, use_gui=False, num_seconds=20000, max_depart_delay=100000,
                    time_to_teleport=-1, delta_time=10, yellow_time=2, min_green=5, max_green=50, single_agent=False):
        # input files
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui

        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')
        
        # parameter setting
        self.sim_max_time = num_seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        # open sumo simulation
        traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        self.single_agent = single_agent
        if net_file == "networks/Cologne/cologne.net.xml":
            self.ts_ids = ['GS_1456583705', 'GS_1456583696', 'GS_73523271', 'GS_1730910809', 'GS_1456583690', 'GS_1456583691', 'GS_1456583679', 'GS_1456583681', 'GS_1694971682']
            self._a_net = "networks/Cologne/vtypes.add.xml"
        else:
            self.ts_ids = traci.trafficlight.getIDList()
            self._a_net = None
        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()  # queue?

        self.last_reward = np.zeros(shape=(len(self.ts_ids)))

        self.rewards = {}
        self.matrix = {}
        self.graph = {}
        self.adj = np.zeros([len(self.ts_ids), len(self.ts_ids)])

        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = ''

        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name

        for junction in self.ts_ids:
            link = []
            self.matrix[junction] = np.zeros(len(self.ts_ids))
            lanes = traci.trafficlight.getControlledLanes(junction)
            for item in enumerate(lanes):
                link.append(item[1].replace('-',''))
            self.graph[junction] = link
        for junction1 in self.ts_ids:
            i = 0
            for junction2 in self.ts_ids:
                a = self.graph[junction1]
                b = self.graph[junction2]
                x = set(a).isdisjoint(set(b))
                if x is False:
                    self.matrix[junction1][i] = 1
                i += 1
        tmp = []
        for junction in self.ts_ids:
            tmp.append(list(self.matrix[junction]))
            self.adj = np.mat(tmp)

        traci.close()

    def reset(self):
        if self.run !=0:
            traci.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        # restart simulation cmd
        if self._a_net == None:
            sumo_cmd = [self._sumo_binary,
                        '-n', self._net,
                        '-r', self._route,
                        '--max-depart-delay', str(self.max_depart_delay),
                        '--waiting-time-memory', '10000',
                        '--time-to-teleport', str(self.time_to_teleport),
                        '--random']
        else:
            sumo_cmd = [self._sumo_binary,
                            '-n', self._net,
                            '-r', self._route,
                            '-a', self._a_net,
                            '--max-depart-delay', str(self.max_depart_delay),
                            '--waiting-time-memory', '10000',
                            '--time-to-teleport', str(self.time_to_teleport),
                            '--begin', '21600',
                            '--random']

        if self.use_gui:
            sumo_cmd.append('--start')

        traci.start(sumo_cmd)

        # reset traffic simulation
        self.traffic_signals = {ts: TrafficSignal(self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green) for ts in self.ts_ids}
        self.vehicles = dict()

        # agent means traffic lights
        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        # return current simulation time on SUMO
        return traci.simulation.getTime()

    def step(self, action):
        rewards = self._compute_rewards()
        self.rewards = rewards
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()   # simulate for one step
    
        else:
            # print("action:", action)
            self._apply_actions(action)

            time_to_act = False
            while not time_to_act:
                self._sumo_step()

                for ts in self.ts_ids:
                    self.traffic_signals[ts].update()
                    if self.traffic_signals[ts].time_to_act:
                        time_to_act = True

                if self.sim_step % 5 == 0:
                    info = self._compute_step_info()
                    self.metrics.append(info)
        
        observations = self._compute_observations()
        done = {'__all__': self.sim_step > self.sim_max_time}
        done.update({ts_id: False for ts_id in self.ts_ids})    #initialize?


        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], done['__all__'],{}
        else:
            return observations, rewards, done, {}

    def test_data(self):
        total_stop = sum(self.traffic_signals[ts].get_total_queue() for ts in self.ts_ids)
        waiting_time = sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        return total_stop,waiting_time

    def _apply_actions(self, actions):
        '''
        Set the next green phase for the trsffic signals
        actions: If single-agent, actions is an int between 0 and self.num_green_phase (next green phase) ---- need to be changed into duration time!!
                 If multiagent, actions is a dict{ts_id: greenPhase}
        '''
        if self.single_agent:
            self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                self.traffic_signals[ts].set_next_phase(action)
        
    def _compute_observations(self):
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids}
    
    def _compute_rewards(self):
        reward = {}
        for ts in self.ts_ids:
            reward[ts] = self.traffic_signals[ts].compute_reward()
        return reward

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    # multi agent
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        time.sleep(0.1)
        traci.simulationStep()
    
    def get_ts_id(self):
        return self.ts_ids

    def compute_waiting_time(self):
        return sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)

    def compute_queue(self):
        return{ts: self.traffic_signals[ts].get_lanes_queue() for ts in self.ts_ids}
        
    def compute_per_waiting_time(self):
        return {ts: sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids}

    def compute_pressure(self):
        return {ts: self.traffic_signals[ts].get_phase_pressure() for ts in self.ts_ids}

    def compute_priority(self):
        return {ts: self.traffic_signals[ts].get_phase_priority() for ts in self.ts_ids}

# output file data
    def _compute_step_info(self):
        info = {}
        for ts in self.ts_ids:
            current_dict = {
            'step_time': self.sim_step,
            'waiting_time': sum(self.traffic_signals[ts].get_waiting_time_per_lane()),
            'queue_length': self.traffic_signals[ts].get_total_queue(),
            'average_speed': sum(self.traffic_signals[ts].get_speed_per_lane())/len(self.traffic_signals[ts].lanes),
            'density': sum(self.traffic_signals[ts].get_lanes_density())/len(self.traffic_signals[ts].lanes),
            'emission': sum(self.traffic_signals[ts].get_emission_per_lane()),
            }
            info[ts] = current_dict
        return info
    
    def close(self):
        traci.close()

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            df.to_csv(out_csv_name + 'run{}'.format(run) + '.csv', index=False)
