from Simulator.src.algorithm_abstract import Algorithm
import numpy as np
import time, copy
from Simulator.utils.monitoring import calculate_cost, sort_requests


def group_containers(a):
    x = np.split(a, np.unique(a[:, 1], return_index=True)[1])[::-1][0]
    x = x.astype(int)
    return x


class BestFitBatch(Algorithm):

    def is_less_than(self, wastage, min_wastage):
        # return np.less(wastage, min_wastage).all()
        # print(f"Wastage: {wastage} & Min wastage: {min_wastage}")
        return wastage < min_wastage

    def calculate_wastage(self,machine, container, max_values):
        # print(f"Wastage:{np.sum((machine[1:5] - container[2:6])/max_values)}")
        return np.prod((machine[1:5] - container[2:6])/max_values)
    

    def __init__(self) -> None:
        self.batch_times = list()

    '''
    __call__ method helps to call objects like function.
    '''
    def __call__(self, cluster, clock, **kwargs):

       
        # Dicts of machines and containers
        containers = cluster.container_which_has_waiting_instance
        l = len(cluster.machines)

        max_values = cluster.max_values

        if len(containers) == 0:
            return 0
        
        start = time.time()

        selected_container = None
        selected_cnt_idx = None 
        selected_mc_idx = None
        flag = False
        bucket_size = 0

        for container in containers:
            cnt_idx = container[0].astype(int)
            min_wastage = 1e9
            selected_container = None
            selected_cnt_idx = None 
            selected_mc_idx = None

            for i in range(l):
                if np.sum(cluster.machines[i, 1:5] >= container[2:6]) == 4:
                    machine = cluster.machines[i]
                    mc_idx = machine[0].astype(int)
                    wastage = self.calculate_wastage(machine, container, max_values)

                    if wastage < min_wastage:
                        min_wastage = wastage
                        selected_container = container
                        selected_cnt_idx = cnt_idx
                        selected_mc_idx = mc_idx
                        flag = True


            cluster.containers_dict[cnt_idx][1]['tried_scheduling'] = True 

            if selected_mc_idx != None:
                cluster.schedule(selected_container, selected_mc_idx)
                bucket_size += 1
                cluster.containers_dict[selected_cnt_idx][1]['tried_scheduling'] = True
            else:
                print("NOT SCHEDULED")
 
        
            
        if flag:   
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

        return 0
