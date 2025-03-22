from collections import defaultdict
import numpy as np
import copy
from collections import defaultdict

class Cluster(object):

    def __init__(self, env, bins = None):

        self.env = env

        # Numpy arrays of machines and containers
        self.machines = list()
        self.machines_cap = list()
        self.containers = np.empty((0,7))

        # Attribute dict of machines and containers
        self.machines_dict = defaultdict(list)
        self.containers_dict = defaultdict(list)
        self.scheduled_idx = 0 # To take next request from list

        # bins
        self.bins = bins

        # This flag helps schduler to stop scheduling in case no more resources are available
        # start when they become free thus preventing repeated calls for decision
        self.can_schedule = True

        # cluster machine assigned to
        self.cluster = None

        # Events and duration
        self.write_ts = -1
        self.events = list()
        self.duration = defaultdict(list)
        self.cummulative_cost = list()
        self.bucket_sizes = list()
        self.ct_mc_map = dict()
        

    def add_machines(self, machines, machines_dict):
        '''
        Machine:
            0 : idx
            1 : cpu
            2 : mem
            3 : disk
            4 : net
            5: cost
        '''

        self.machines = machines
        self.machines_cap = copy.deepcopy(machines)
        self.max_values = self.machines_cap[:,1:5].max(axis = 0)
        self.machines_dict = machines_dict

    def add_container(self, container, container_name):
        '''
        Container:
            0 : idx
            1 : start_ts
            2 : cpu
            3 : mem
            4 : disk
            5 : net
            6 : duration
        '''
        
        # print(self.containers.shape, container.shape)
        self.containers = np.append(self.containers, container)
        cnt_idx = container[0].astype(int)

        self.containers = self.containers.reshape(-1, np.size(container,0))

        container_data = {
            'arrival' : container[1].astype(int),
            'process' : None,
            'started' : False,
            'finished' : False,
            'start_ts' : None,
            'finish_ts': None,
            'machine' : None,
            'tried_scheduling' : False,
        }
        self.containers_dict[cnt_idx] = [container_name, container_data]

    
    def run_container(self, container, mc_idx):
        cnt_idx = container[0].astype(int)

        if np.isnan(container[6]) or container[6] < 0:
            yield self.env.timeout(float('inf'))
        else:
            yield self.env.timeout(container[6])

        self.can_schedule = True
        self.stop_container(container, mc_idx)
        self.containers_dict[cnt_idx][1]['finished'] = True
        self.containers_dict[cnt_idx][1]['finish_ts'] = self.env.now
        


    
    def schedule(self, container, mc_idx):

        self.scheduled_idx += 1
        cnt_idx = container[0].astype(int)
        self.containers_dict[cnt_idx][1]['started'] = True
        self.containers_dict[cnt_idx][1]['start_ts'] = self.env.now
        self.containers_dict[cnt_idx][1]['machine'] = mc_idx
        
        # START CONTAINER
        self.start_container(container, mc_idx)

        self.containers_dict[cnt_idx][1]['process'] = self.env.process(self.run_container(container, mc_idx))

        # Save container to machine mapping
        self.ct_mc_map[container[0]] = mc_idx

        # SAVE DURATION
        self.save_duration(mc_idx, self.env.now, self.env.now + container[-1])
        
    def start_container(self, container, mc_idx):
        self.machines[mc_idx][1:5] -= container[2:6]


    def stop_container(self, container, mc_idx):
        self.machines[mc_idx][1:5] += container[2:6]

    
    def save_duration(self, mid, start_time, end_time):
        if self.duration.get(mid) == None:
            self.duration[mid].append([start_time, end_time])
        else:
            old_end = self.duration[mid][-1][1]
            if start_time > old_end:
                self.duration[mid].append([start_time, end_time])

            elif start_time < old_end and end_time > old_end:
                self.duration[mid][-1][1] = end_time

            

    @property
    def has_waiting_container_instances(self):
        return len(self.containers) > self.scheduled_idx

    @property
    def container_which_has_waiting_instance(self):
        idxs = []
        if self.has_waiting_container_instances:
            idxs = [idx for idx, (key, value) in enumerate(self.containers_dict.items()) if value[1]['started'] == False]
            return self.containers[idxs].astype(int)
        return idxs

    @property
    def finished_container(self):
        lst = []
        for cid, data in self.containers_dict.values():
            if data['finished']:
                lst.append(cid)     
        return lst 
    
    @property
    def unfinished_container(self):
        lst = []
        for cid, data in self.containers_dict.values():
            if not data['finished']:
                lst.append(cid)      
        return lst 


    @property
    def finished(self):
        if self.has_waiting_container_instances:
            return False
        return True