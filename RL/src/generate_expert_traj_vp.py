from pyvpsolver.solvers import mvpsolver
from collections import deque, defaultdict 

import numpy as np
from tqdm import tqdm
from copy import deepcopy

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

    for machine in machines:
        x = machine.astype(int)
        bin_to_machine_maps[tuple(x[1:5])].append(x[0])

    cost = bin_capacities[:,0]*9 + bin_capacities[:,1]

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


def vp_mappings(machines, containers, grp_ct):

    actions_dict = {}
    if np.size(containers, 0) == 0:
        return actions_dict
    
    machines_cap = deepcopy(machines)
    
    splits = list(range(0,len(containers), grp_ct))
    start = splits[0]
    if len(splits) <= 1:
        end = start

    for i in tqdm(range(1, len(splits))):
        end = splits[i]
        grouped_containers = containers[start:end,:]

        if np.size(grouped_containers, 0) == 0:
            print("Size grp 0")
            return actions_dict
        
        data = generate_data(machines, grouped_containers, machines_cap)

        solution = None

        try:
            solution = mvpsolver.solve(
                data['bin_capacities'], data['cost'], data['num_bins'],
                data['item_capacities'], data['demand'],
                script='vpsolver_gurobi.sh',
                verbose=False,
                script_options="Threads=12",    
            )

            # mvpsolver.print_solution(solution)

        except Exception as e:
            neg_m = (machines < 0).any(axis=1).any()
            print("EXCEPTION", e, neg_m)
            start = end
            continue


        # lst: 
        # [
        #       (1, [(0, 0), (3, 0), (9, 0)]), 
        #       (1, [(1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)])
        # ]
        # print(data['bin_capacities'], data['mids'])
        objective, lst_sol = solution
        # print(data['cids'])
        # print(lst_sol)
        container_set = set()
        for idx, sol in enumerate(lst_sol):
            x = tuple(data['bin_capacities'][idx])
            mc_idx = 0
            for multiplier, pattern in sol:
                for i in range(multiplier):
                    schedule_mc_idxs = data['mids'][x]
                    for it, opt in pattern:
                        schedule_ct_idx = data['cids'][it]
                        schedule_mc_idx = schedule_mc_idxs[mc_idx]
                        container_set.add(schedule_ct_idx)
                        assert schedule_ct_idx == grouped_containers[it][0]

                        actions_dict[grouped_containers[it][0]] = schedule_mc_idx
                        # print("B4=",schedule_mc_idx, machines[schedule_mc_idx, 1:])
                        machines[schedule_mc_idx, 1:5] -= grouped_containers[it][2:6]
                        # print("After=",schedule_mc_idx, machines[schedule_mc_idx, 1:])
                        print("Scheduling %s on %s " %(grouped_containers[it][0], schedule_mc_idx) )
                        # print(schedule_mc_idx, machines[schedule_mc_idx, 1:])
                        # input()
                mc_idx += 1
        start = end

    

    # final batch
    grouped_containers = containers[start:,:]
    data = generate_data(machines, grouped_containers, machines_cap)

    solution = None
    try:
        solution = mvpsolver.solve(
            data['bin_capacities'], data['cost'], data['num_bins'],
            data['item_capacities'], data['demand'],
            script='vpsolver_gurobi.sh',
            verbose=False,
            script_options="Threads=12",    
        )

        # mvpsolver.print_solution(solution)

    except Exception as e:
        neg_m = (machines < 0).any(axis=1).any()
        print("EXCEPTION", e, neg_m)
        return actions_dict
    

    objective, lst_sol = solution
    container_set = set()
    for idx, sol in enumerate(lst_sol):
        x = tuple(data['bin_capacities'][idx])
        mc_idx = 0
        for multiplier, pattern in sol:
            for i in range(multiplier):
                schedule_mc_idxs = data['mids'][x]
                for it, opt in pattern:
                    schedule_ct_idx = data['cids'][it]
                    schedule_mc_idx = schedule_mc_idxs[mc_idx]
                    container_set.add(schedule_ct_idx)

                    assert schedule_ct_idx == grouped_containers[it][0]
                    actions_dict[grouped_containers[it][0]] = schedule_mc_idx
                    # print("B4=",schedule_mc_idx, machines[schedule_mc_idx, 1:])
                    machines[schedule_mc_idx, 1:5] -= grouped_containers[it][2:6]
                    # print("After=",schedule_mc_idx, machines[schedule_mc_idx, 1:])
                    print("Scheduling %s on %s " %(grouped_containers[it][0], schedule_mc_idx) )
            mc_idx += 1

    
    return actions_dict
    



