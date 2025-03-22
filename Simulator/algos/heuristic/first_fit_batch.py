from Simulator.src.algorithm_abstract import Algorithm
import time, copy
import numpy as np
from Simulator.utils.monitoring import calculate_cost, sort_requests, get_utlization, print_active_machines

def group_containers(a):
    x = np.split(a, np.unique(a[:, 1], return_index=True)[1])[::-1][0]
    x = x.astype(int)
    return x


class FirstFitBatch(Algorithm):

    def __init__(self) -> None:
        self.batch_times = list()
        self.scheduling_step = 0
    '''
    __call__ method helps to call objects like function.
    '''
    def __call__(self, cluster, clock, **kwargs):

        l = len(cluster.machines)

        containers = cluster.container_which_has_waiting_instance
        bucket_size = 0

        if len(containers) == 0:
            return 0
        
        start = time.time()

        flag = False 

        # print("Before=",containers, containers.shape)
        containers = sort_requests(containers, reverse=True)

        # For each item in the batch
        # schedule for the first available bin
        for container in containers:
            cnt_idx = container[0].astype(int)
            # flag = False
            for i in range(l): 
                if np.sum(cluster.machines[i, 1:5] >= container[2:-1]) == 4:
                    machine = cluster.machines[i]
                    mc_idx = machine[0].astype(int)
                    
                    cluster.schedule(container, mc_idx)
                    cluster.containers_dict[cnt_idx][1]['tried_scheduling'] = True
                    flag = True
                    bucket_size += 1
                    break
            if not flag:
                print("NOT SCHEDULED")
                print(f"cnt_idx: {cnt_idx}, ts: {container[1]}")
            cluster.containers_dict[cnt_idx][1]['tried_scheduling'] = True

        if not flag:
            return 0
        
        end = time.time()
        self.batch_times.append(end-start)

        
        cluster.write_ts = int(containers[-1][1])
        cluster.events.append([cluster.write_ts, copy.deepcopy(cluster.machines)])
        cluster.bucket_sizes.append(bucket_size)
        
        # CUMMULATIVE COSTS
        if cluster.bins:
            cost = calculate_cost(cluster.duration, cluster.bins) / 3600
            cluster.cummulative_cost.append(cost)

        return 1
 