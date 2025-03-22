from Simulator.src.algorithm_abstract import Algorithm
import os, gym, copy, time, sys
import numpy as np
import torch as th
from collections import defaultdict, deque

import gym_packing
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from Simulator.utils.monitoring import calculate_cost, sort_requests

class prettyDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self,*args,**kwargs)

    def __repr__(self):
        return str(dict(self))
    

def mask_fn(env: gym.Env) -> np.ndarray:
        # Use this function to return the action masks
        return env.valid_action_mask() 


class PPOBinPack(Algorithm):
    
    def __init__(self, config, env_name, model_path):
        self.env = gym.make(env_name, env_config=config)
        self.max_vals = config['max_vals']
        self.env = ActionMasker(self.env, mask_fn)  # Wrap to enable masking
        self.model = MaskablePPO.load(model_path, env=self.env, device='cpu')

        self.batch_times = list()

    def normalize_data(self, data, type='machines'):

        if type == 'machines':
            machines = data[0].astype(int)
            machines_cap = data[1].astype(int)
            mask = np.any(machines != machines_cap, 1)

            machines_norm = machines[:,1:5] / self.max_vals
            machines_norm = np.c_[mask.reshape(-1,1), machines_norm]
            return machines_norm

        elif type == 'containers':
            data = data / self.max_vals
            data = np.c_[np.zeros(np.size(data, 0)), data]
            return data
        
    
    def __call__(self, cluster, clock, **kwargs):

        containers = cluster.container_which_has_waiting_instance

        if np.size(containers, 0) == 0:
                return 0
        
        grouped_containers = containers

        flag = False
        tic = time.time()
        demands = self.normalize_data(grouped_containers[:,2:6], type='containers')
        bucket_size = 0

        for idx, container in enumerate(grouped_containers):
            cnt_idx = container[0].astype(int)
            
            machines_avail = self.normalize_data([cluster.machines, cluster.machines_cap], type='machines')
            request_norm = demands[idx]
            
            obs = np.vstack(
                [
                    machines_avail, # pm_on/off, cpu, mem, disk, net
                    request_norm.reshape(1,-1), # dummy, cpu, mem, disk, net
                ]
            ).astype(np.double)
            
            self.env.set_obs(obs, cnt_idx)
            action_masks = mask_fn(self.env)

            # Predicitons: action and next hidden state to be used in recurrent policies
            m_idx, _next_hidden_state = self.model.predict(obs, action_masks=action_masks)
            m_idx = int(m_idx)

            if np.sum(cluster.machines[m_idx, 1:5] >= container[2:-1]) == 4:
                cluster.schedule(container, m_idx)
                cluster.containers_dict[cnt_idx][1]['tried_scheduling'] = True
                bucket_size += 1
                flag = True
            else:
                print("NOT SCHEDULED")
                cluster.containers_dict[cnt_idx][1]['tried_scheduling'] = True

        
        if not flag:
            return 0

        toc = time.time()
        self.batch_times.append(toc-tic)
        print(f"Batch time: {toc-tic:0.4f}")
        cluster.write_ts = int(grouped_containers[-1][1])
        cluster.events.append([cluster.write_ts, copy.deepcopy(cluster.machines)])
        cluster.bucket_sizes.append(bucket_size)

        # CUMMULATIVE COSTS
        if cluster.bins:
            cost = calculate_cost(cluster.duration, cluster.bins) / 3600
            cluster.cummulative_cost.append(cost)
        
        return 1