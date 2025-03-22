import numpy as np
from collections import defaultdict
import copy
import gym
from gym import spaces
from gym_packing.utils import *

BIG_NEG_REWARD = -1000

class VMPackingEnvBCRealDuration(gym.Env):
    '''
    Observation:
        Type: Tuple, Discrete
        [0][:, 0]: Binary indicator for open VM's
        [0][:, 1]: CPU load of VM's
        [0][:, 2]: Memory load of VM's
        [0][:, 3]: Disk load of VM's
        [0][:, 4]: Network load of VM's
        [1][0]: Dummy value to match dimension with VMs
        [1][1]: Current CPU demand
        [1][2]: Current memory demand
        [1][3]: Current disk demand
        [1][4]: Current network demand

    Actions:
        Type: Discrete
        Integer of VM number to send VM to that VM

    Reward:
        Negative of the waste, which is the difference between the current
        size and excess space on the VM.

    Starting State:
        No open VM's and first request
        
    Episode Termination:
        When invalid action is selected, attempt to overload VM, or step
        limit is reached.
    '''

    def __init__(self,  *args, **kwargs):
        # Fill with dummy values that can be overridden by env_config
        self.eps = 1e-6
        self.step_limit = 75 # number of requests
        self.n_pms = 15 # number of machines

        self.machines = np.empty(shape=(self.n_pms,4), dtype=np.double) # cpu, mem, disk, net # INT
        self.max_vals = np.empty(shape=(1,4),dtype=np.double) # max values of cpu mem disk net
        self.shuffle_cnt = False
        self.shuffle_mc = False
        self.train = True
        self.containers_org = np.empty(shape=(self.step_limit, 7),dtype=np.double) # id, ts, cpu, mem, disk, net, duration
        self.cost = np.array([1,2,3,4])
        assign_env_config(self, kwargs)

        self.containers = None
        # if len(self.containers_org) == 1000: # where we have only 1000 containers use shuffling
        self.containers = self.containers_org
        self.machines_cap_int = copy.deepcopy(self.machines) # deepcopy of machines
        self.machines_cap_norm = self.normalize_data(self.machines) # vm_on/off, cpu, mem, disk, net
        self.demand = None # dummy, cpu, mem, disk, net # (Norm)

        self.active_machines = 0
        self.current_step = 0

        self.start_idx = 0 
        self.end_idx = 1000
        self.one_round_done = False
        self.action_space = spaces.Discrete(self.n_pms) # Choose one of the machine
        self.observation_space = spaces.Box(low = 0, high = 1, shape=(self.n_pms+1, 5), dtype=np.double)

        # print(self.observation_space.shape)

        self.state = self.reset()

    def normalize_data(self, data, type='machines'):

        if type == 'machines':
            data = data / self.max_vals
            data = np.c_[np.zeros(np.size(data, 0), dtype=np.double), data]
        else:
            data = data / self.max_vals
            data = np.c_[np.zeros(np.size(data, 0), dtype=np.double), data]
        return data

    def update_instance(self, mc, ct):
        ct_norm = ct / self.max_vals
        mc_norm = mc / self.max_vals
        return mc_norm - ct_norm, ct_norm
    
    def reset(self):
        self.current_step = 0 # signifies the current process
        self.active_machines = 0
        self.vm_mc_mapping = {} # dict to map vm req number to pm number
        self.reward_list = list()

        if self.shuffle_cnt == True: 
            np.random.shuffle(self.containers)

        self.step_limit = len(self.containers)

        if self.shuffle_mc == True:
            np.random.shuffle(self.machines)

        
        if self.train:
            self.end_time_set = set()
            self.end_time_map = defaultdict(list)
        
        self.machines_avail = copy.deepcopy(self.machines_cap_norm) # vm_on/off, cpu, mem, disk, net # (Norm)
        # self.machines_cap_norm_sum = self.machines_avail[:, 1:].sum(axis = 1) # vm_on/off, cpu, mem, disk, net # (Norm)

        # cpu, mem, disk, net
        self.machines_avail_int = copy.deepcopy(self.machines) # deepcopy of machines 
        self.containers_int = copy.deepcopy(self.containers[:,2:-1]) # deep copy of containers
        self.demand = self.normalize_data(self.containers_int, type='containers')
        self.durations = copy.deepcopy(self.containers[:,-1]) # duration
        self.arrival_time = copy.deepcopy(self.containers[:,1]) # ts

        # print("Containers max=",self.demand.max(axis=0))
        request_norm = self.demand[self.current_step]
        
        self.state = np.vstack(
            [
                self.machines_avail, # pm_on/off, cpu, mem, disk, net
                request_norm.reshape(1,-1), # dummy, cpu, mem, disk, net
            ]
        ).astype(np.double)

        return self.state

    def step(self, action):
        done = False

        pm_state = self.state[:-1]
        demand = self.state[-1]

        reward = 0

        # Replinish - START
        end_time = self.arrival_time[self.current_step] + self.durations[self.current_step]
        arrival_time = self.arrival_time[self.current_step]

        # If there are tasks that leave at current tasks arrival time or before that
        # Get those tasks and relinquish resources to respective VMs
        prev_task_end_times = list(filter(lambda x: x <= arrival_time + 1, self.end_time_set))

        for et in prev_task_end_times:
            # get vm id and task id
            for vm_id, task_id in self.end_time_map[et]:
                pm_state[vm_id, 1:] += self.demand[task_id, 1:]
                self.machines_avail_int[vm_id] += self.containers_int[task_id]
                
                # Shut down VM if no ms active on it
                if np.all(self.machines_avail_int[vm_id] == self.machines_cap_int[vm_id]):
                    pm_state[vm_id] = self.machines_cap_norm[vm_id]
                    self.active_machines -= 1

            # remove this end_time time from set
            self.end_time_set.remove(et)

        # Add current task to map
        self.end_time_map[end_time].append([action, self.current_step])
        self.end_time_set.add(end_time)

        # Replinish - END
        if action < 0 or action >= self.n_pms:
            raise ValueError("Invalid action: {}".format(action))

        # Demand doesn't fit into PM
        elif any(pm_state[action, 1:] - demand[1:] < 0 - self.eps):
            reward = BIG_NEG_REWARD
            done = True
            print(f"Does not fit step={self.current_step}, action={action}, {pm_state[action, 1:] - demand[1:]}, reward={reward}")
            print(f"Mc: {self.machines_avail_int[action], self.containers_int[self.current_step]}")
            input()
        else:
            if pm_state[action, 0] == 0:
                # Open PM if closed
                pm_state[action, 0] = 1
                self.active_machines += 1

            # Reduce the resources
            pm_state[action, 1:] -= demand[1:]
            self.machines_avail_int[action] -= self.containers_int[self.current_step]

            reward = self.calculate_reward_on_space(pm_state)
            
        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True
        self.update_state(pm_state)
 
        return self.state, reward, done, {'active_machines':self.active_machines, 'reward':reward}



    def update_state(self, pm_state):

        # Make action selection impossible if the PM would exceed capacity
        step = self.current_step if self.current_step < self.step_limit else self.step_limit-1
        self.machines_avail = self.normalize_data(self.machines_avail_int)

        # print(self.machines_avail.shape, pm_state.shape)
        self.machines_avail[:,0] = pm_state[:,0]

        data_center = np.vstack([
            self.machines_avail, 
            self.demand[step].reshape(1,-1)]
        ).astype(np.double)

        data_center = np.where(data_center>1,1, data_center) # Fix rounding errors
        self.state = data_center

    def calculate_reward_on_space(self, pm_state):
        # reward = -np.sum(pm_state[:, 0] * pm_state[:,1])
        active_cost =  np.sum(pm_state[:, 0] * self.cost)
        reward = - active_cost / np.sum(self.cost)
        return reward
    
    def sample_action(self):
        return self.action_space.sample()

    def valid_action_mask(self):
        pm_state = self.state[:-1]
        demand = self.state[-1]
        action_mask = (pm_state[:,1:] - demand[1:]) > 0 - self.eps
        action_mask1 = (action_mask.sum(axis=1)==4).astype(int)
        return action_mask1 

    def action_masks(self):
        return self.valid_action_mask()

    def get_obs(self):
        return self.state 

    def set_obs(self, state, cidx):
        self.state = state
        self.current_step = cidx 
