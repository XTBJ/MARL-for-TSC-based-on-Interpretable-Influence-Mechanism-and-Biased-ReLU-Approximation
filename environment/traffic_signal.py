'''
traffic signals setting
'''
import os
import queue
import sys
from typing import Tuple
import time

from gym.spaces import space

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
    
import traci
from traci._trafficlight import Logic
import numpy as np
from gym import spaces

class TrafficSignal:
    '''
    This class represent a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase (duration time) using Traci API
    '''

    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = 0
        self.last_measure = 0.0
        self.waiting_time_last_measure = 0.0
        self.density_last_measure = 0.0
        self.halting_number_last_measure = 0.0
        self.last_reward = None
        self.phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        self.num_green_phases = len(self.phases) // 2   # number of green phases == number of phases (green+yellow)/2   num(green_phases) = num(action)
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        self.out_lanes = [link[0][1] for link in traci.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))  # duplicate removal -- order may sort()?
        self.signal_counts = [traci.lane.getLinkNumber(lane) for lane in self.lanes]

        lanes_phases_mapping = [lane for lane, count in zip(self.lanes, self.signal_counts) for _ in range(count)]

        self.lanes_signals = {}

        for phase in self.phases:
            incoming_lanes = set()
            outgoing_lanes = set()
            outgoing_lanes_list = []
            
            current_signal = phase.state
            traci.trafficlight.setRedYellowGreenState(self.id, current_signal)

            if 'G' in current_signal:
                # print('num_crrent_signal:', len(current_signal))
                # for lane, count in zip(self.lanes, self.signal_counts):
                for i, s in enumerate(current_signal):
                    if s == 'G':
                        # set
                        incoming_lanes.add(lanes_phases_mapping[i])
                        links = traci.lane.getLinks(lanes_phases_mapping[i])
                        outgoing_lanes_list.append([link[0] for link in links if link[1] == True])
                for sublist in outgoing_lanes_list:
                    for item in sublist:
                        outgoing_lanes.add(item)

                self.lanes_signals[current_signal] = {
                    'incoming_lanes': list(incoming_lanes),
                    'outgoing_lanes': list(outgoing_lanes)
                }
                    
        '''
        Default observation space is a vector: R^(greenPhases + 2*lanes)
        s = [current phase one-hot encoded, density for each lane, queue for each lane] 
        Can change this by modifing self.observation_space and the method_compute_observations()

        Action space is which green phase is going to be open for the next delta_time seconds.  // change to be duration
        '''
        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases + 2*len(self.lanes)), high=np.ones(self.num_green_phases + 2*len(self.lanes)))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                     # Green Phase
            *(spaces.Discrete(10) for _ in range(2*len(self.lanes)))    # Density and stopped-density for each lane //why 10?
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

        logic = Logic("new-program"+self.id, 0, 0, phases=self.phases)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)

    @property
    def phase(self):
        return traci.trafficlight.getPhase(self.id)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:    # reach yellow_time end
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.is_yellow = False

# If action is duration, change this function
    def set_next_phase(self, new_phase):
        '''
        Sets what will be the next green phase and set yellow phase id the next phase is different than the current
        new_phase: (int) Number between [0..num_green_phases] --- a vector? choose the phase
        '''
        # yellow light phase!
        # difference between phase and action
        new_phase *=2
        if self.phase == new_phase or self.time_since_last_phase_change < self.min_green + self.yellow_time:
            self.green_phase = self.phase
            traci.trafficlight.setPhase(self.id, self.green_phase)
            self.next_action_time = self.env.sim_step + self.delta_time     #cumulative
        else:
            self.green_phase = new_phase
            traci.trafficlight.setPhase(self.id, (self.phase + 1)%self.num_green_phases)    # turn yellow --- next phase is different, turn yellow firstly
            self.next_action_time = self.env.sim_step + self.delta_time + self.yellow_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        phase_id = [1 if self.phase//2 == i else 0 for i in range(self.num_green_phases)]   # same as action
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        waiting_time = self.get_waiting_time_per_lane()
        observation = np.array(phase_id + density + queue)
        return observation

    def compute_reward(self):
        self.last_reward = self.multi_reward()        
        return self.last_reward
    
    def _pressure_reward(self):
        return -self.get_pressure()

    def _queue_average_reward(self):
        new_average = np.mean(self.get_stopped_vehicles_num())
        reward = self.last_measure - new_average    # more cars, reward will be negative
        self.last_measure = new_average
        return reward

    def _queue_reward(self):
        return -(sum(self.get_lanes_queue()))**2   # square

    def _my_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        halting_number = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

        if (halting_number - self.last_measure)<0:
            reward = -5*(halting_number - self.last_measure) 
        else:
            reward = -ts_wait/5
        self.last_measure = halting_number

        self.waiting_time_last_measure = ts_wait
        
        if halting_number == 0:
            reward = reward + 25
        return reward

    def single_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        halting_number = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        if halting_number == 0:
            reward = reward + 1
        return reward

    def multi_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        halting_number = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

        # halting_number reduce
        if (halting_number - self.last_measure)<0:
            reward = -5*(halting_number - self.last_measure) 
        else:
            reward = -ts_wait/5

        self.last_measure = halting_number
        
        if halting_number == 0:
            reward = reward + 25
        return reward

    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = self.waiting_time_last_measure - ts_wait        # +: cars decrease, -: cars increase 
        self.waiting_time_last_measure = ts_wait
        return reward

    def _density_reward(self):
        ts_den = sum(self.get_lanes_density())
        reward = self.density_last_measure - ts_den
        self.density_last_measure = ts_den
        return reward

    def _halting_number_reward(self):
        ts_hal = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])
        reward = self.halting_number_last_measure - ts_hal
        self.halting_number_last_measure = reward
        return reward

    def _waiting_time_reward2(self):
        ts_wait = self.get_waiting_time()
        self.last_measure = ts_wait
        THR = 50
        if ts_wait < THR:
            reward = (ts_wait-THR)*(ts_wait-THR)
        else:
            #reward = 1.0/ts_wait
            reward = -(ts_wait-THR)*(ts_wait-THR)
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward
    
    def get_waiting_time(self):
        all_vehicle_id = traci.vehicle.getIDList()
        all_vehicle_waiting_time = [(i, traci.vehicle.getWaitingTime(i)) for i in all_vehicle_id]
        Waiting_time = 0.0
        for vehicle_waiting_time in all_vehicle_waiting_time:
            Waiting_time += vehicle_waiting_time[1]
        return Waiting_time


    def get_waiting_time_per_lane(self):
        wait_time_per_lane = [] # a list
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            # wait_time for the selected lane
            wait_time = 0.0
            # for cars in the selected lane
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc =  traci.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane+"wt": acc}
                else:
                    self.env.vehicles[veh][veh_lane+"wt"] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane+"wt"]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_speed_per_lane(self):
        speed_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            speed = 0.0
            count = 0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getSpeed(veh)
                speed += acc
                count += 1
            if count == 0:
                speed = 13.89
            else:
                speed = speed/count
            speed_per_lane.append(speed)
        return speed_per_lane

    def get_emission_per_lane(self):
        emission_per_lane = []
        for lane in self.lanes:
            veh_list = traci.lane.getLastStepVehicleIDs(lane)
            emission = 0.0
            for veh in veh_list:
                veh_lane = traci.vehicle.getLaneID(veh)
                acc = traci.vehicle.getCO2Emission(veh)
                emission += acc
            emission_per_lane.append(emission)
        return emission_per_lane


    def get_pressure(self):
        # Can there be a negative number?
        return abs(sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) - sum(traci.lane.getLastStepVehicleNumber(lane)for lane in self.out_lanes))

    def get_phase_pressure(self):
        phase_pressure = {}
        for i, current_phase in enumerate(self.lanes_signals.keys()):
            veh_in = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes_signals[current_phase]['incoming_lanes'])
            veh_out = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes_signals[current_phase]['outgoing_lanes'])
            phase_pressure[i] = veh_in - veh_out
        return phase_pressure  

    def get_phase_priority(self):
        phase_priority = {}       
        for i, current_phase in enumerate(self.lanes_signals.keys()):
            veh_num = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes_signals[current_phase]['incoming_lanes'])

            veh_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in self.lanes_signals[current_phase]['incoming_lanes'])

            veh_emission = sum(traci.lane.getCO2Emission(lane) for lane in self.lanes_signals[current_phase]['incoming_lanes'])

            # phase_priority[i] = veh_num  + veh_waiting_time * 0.1 + veh_emission * 0.001
            phase_priority[i] = veh_num

        return phase_priority

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize)+2.5(minGap)
        # when traci.lane.getLastStepVehicleNumber(lane) = traci.lane.getLength(lane)*vehicle_size_min_gap
        # reaches critical saturation
        return [min(1,traci.lane.getLastStepVehicleNumber(lane)/(traci.lane.getLength(lane)*vehicle_size_min_gap)) for lane in self.out_lanes]
    
    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize)+2.5(minGap)
        return [min(1,traci.lane.getLastStepVehicleNumber(lane) / (traci.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        return [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]

    def get_edge_queue(self):
        queue = [traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes]
        edge = [sum(queue[i:i+3]) for i in range(0, len(queue), 3)]

    def get_total_queue(self):
        return sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += traci.lane.getLastStepVehicleIDs(lane)
        return veh_list
    