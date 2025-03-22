from Simulator.src.algorithm_abstract import Algorithm

from pyvpsolver.solvers import mvpsolver
from collections import deque, defaultdict 

from Simulator.utils.monitoring import calculate_cost, sort_requests, get_utlization, print_active_machines

import numpy as np
import time, copy


class prettyDict(defaultdict):
    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self,*args,**kwargs)

    def __repr__(self):
        return str(dict(self))


def generate_data(machines, grouped_containers, machines_cap):
    total_ct_grp = np.size(grouped_containers, 0)

    bin_to_machine_maps = prettyDict(deque)
    data = {}

    # Bins
    bin_capacities, indices, num_bins = np.unique(machines[:, 1:5], return_index=True, return_counts=True, axis = 0)

    bin_capacities = bin_capacities.astype(int)

    cost = list()

    for machine in machines:
        x = machine.astype(int)
        bin_to_machine_maps[tuple(x[1:5])].append(x[0])

    cost = machines_cap[indices,-1] # used for alibaba & google



    demand = [1 for _ in range(total_ct_grp)]
    cids = grouped_containers[:,0]
    item_capacities = [[x] for x in grouped_containers[:,2:6]]

    data['mids'] = bin_to_machine_maps
    data['bin_capacities'] = bin_capacities
    data['cost'] = cost
    data['num_bins'] = num_bins

    data['cids'] = cids 
    data['item_capacities'] = item_capacities
    data['demand'] = demand
    return data

def group_containers(a):
    x = np.split(a, np.unique(a[:, 1], return_index=True)[1])[::-1][0]
    x = x.astype(int)
    return x

class VpSolverBatch(Algorithm):

    def __init__(self, optimizer):
        self.optimizer = 'vpsolver_gurobi.sh' if optimizer == 'GUROBI' else 'vpsolver_scip.sh'
        self.batch_times = list()
        self.scheduling_step = 0

    '''
    __call__ method helps to call objects like function.
    '''
    def __call__(self, cluster, clock, **kwargs):

        # print(cluster.can_schedule)
        if cluster.can_schedule:

            start_time = time.time()

            # Machine and container np array
            machines = cluster.machines

            machines_cap = cluster.machines_cap

            containers = cluster.container_which_has_waiting_instance

            total_containers = np.size(containers, 0)
            if total_containers == 0:
                return 0

            
            start, end = 0, 0
            splits = list(range(0,total_containers, 16))
            start = splits[0]

            if len(splits) <= 1:
                end = start

            for i in range(1, len(splits)):
                end = splits[i]
                grouped_containers = containers[start:end, :]
                if np.size(grouped_containers, 0) == 0:
                    return 0
                
                machines = cluster.machines
                machines_cap = cluster.machines_cap

                m_dict = cluster.machines_dict
                c_dict = cluster.containers_dict

                bucket_size = 0

                data = generate_data(machines, grouped_containers, machines_cap)
                solution = None

                solution = mvpsolver.solve(
                    data['bin_capacities'], data['cost'], data['num_bins'],
                    data['item_capacities'], data['demand'],
                    script=self.optimizer,
                    verbose=False,
                    script_options="Threads=12",    
                    )
                
                    # mvpsolver.print_solution(solution)
                

                # lst: 
                # [
                #       (1, [(0, 0), (3, 0), (9, 0)]), 
                #       (1, [(1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)])
                # ]
                objective, lst_sol = solution
                container_set = set()
                for idx, sol in enumerate(lst_sol):
                    x = tuple(data['bin_capacities'][idx])
                    
                    mc_idx = 0

                    for multiplier, pattern in sol:

                        for i in range(multiplier):

                            schedule_mc_idxs = data['mids'][x]
                            schedule_mc_name = m_dict[schedule_mc_idxs[mc_idx]]
                            
                            for it, opt in pattern:
                                schedule_ct_idx = data['cids'][it]
                                schedule_ct_name = c_dict[schedule_ct_idx][0]
                                
                                container_set.add(schedule_ct_idx)
                                # Container, mid
                                cluster.schedule(grouped_containers[it], schedule_mc_idxs[mc_idx])
                                c_dict[schedule_ct_idx][1]['tried_scheduling'] = True 
                                bucket_size += 1
                        mc_idx += 1

                for machine in machines:
                    if (machine<0).any(axis=0).any():
                        print("Infeasible=",machine)
                start = end

            # final batch
            grouped_containers = containers[start:,:]
            if np.size(grouped_containers, 0) > 0:
                machines = cluster.machines
                machines_cap = cluster.machines_cap
                data = generate_data(machines, grouped_containers, machines_cap)

                m_dict = cluster.machines_dict
                c_dict = cluster.containers_dict

                bucket_size = 0

                data = generate_data(machines, grouped_containers, machines_cap)
                solution = None

                # try:
                solution = mvpsolver.solve(
                    data['bin_capacities'], data['cost'], data['num_bins'],
                    data['item_capacities'], data['demand'],
                    script=self.optimizer,
                    verbose=False,
                    script_options="Threads=12",    
                    )

                # lst: 
                # [
                #       (1, [(0, 0), (3, 0), (9, 0)]), 
                #       (1, [(1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)])
                # ]
                # print(data['bin_capacities'], data['mids'])
                objective, lst_sol = solution
                container_set = set()
                for idx, sol in enumerate(lst_sol):
                    x = tuple(data['bin_capacities'][idx])
                    
                    mc_idx = 0

                    for multiplier, pattern in sol:

                        for i in range(multiplier):

                            schedule_mc_idxs = data['mids'][x]
                            schedule_mc_name = m_dict[schedule_mc_idxs[mc_idx]]
                            
                            for it, opt in pattern:
                                schedule_ct_idx = data['cids'][it]
                                schedule_ct_name = c_dict[schedule_ct_idx][0]
                                
                                container_set.add(schedule_ct_idx)
                                # Container, mid
                                cluster.schedule(grouped_containers[it], schedule_mc_idxs[mc_idx])
                                c_dict[schedule_ct_idx][1]['tried_scheduling'] = True 
                                bucket_size += 1
                        mc_idx += 1
                for machine in machines:
                    if (machine<0).any(axis=0).any():
                        print("Infeasible Last=",machine)

            if len(container_set) != np.size(grouped_containers, 0):
                for cid in grouped_containers[:,0]:
                    if cid not in container_set:
                        c_dict[cid][1]['tried_scheduling'] = True 
                print("NOT SCHEDULED")
                return 0
            

            end_time = time.time()
            self.batch_times.append(end_time-start_time)
            print(f"Batch time: {end_time-start_time:0.4f}")
            cluster.write_ts = int(grouped_containers[-1][1])
            cluster.events.append([cluster.write_ts, copy.deepcopy(cluster.machines)])
            cluster.bucket_sizes.append(bucket_size)

            # CUMMULATIVE COST
            if cluster.bins:
                cost = calculate_cost(cluster.duration, cluster.bins) / 3600
                cluster.cummulative_cost.append(cost)

            return 1
        
        return 0
        
