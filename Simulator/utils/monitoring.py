import pickle, os
import numpy as np
import pathlib
from collections import defaultdict, Counter

# Change this path to the path of the monitoring folder
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'MonitoringLogs')
os.makedirs(ROOT, exist_ok=True)

STATE_PATH = os.path.join(ROOT, 'cluster_state')
os.makedirs(STATE_PATH, exist_ok=True)
MACHINE_CT_PATH = os.path.join(ROOT, 'machine_ct')
os.makedirs(MACHINE_CT_PATH, exist_ok=True)
DURATION_PATH = os.path.join(ROOT, 'duration')
os.makedirs(DURATION_PATH, exist_ok=True)
BATCH_TIME_PATH = os.path.join(ROOT, 'batch_time')
os.makedirs(BATCH_TIME_PATH, exist_ok=True)
EXE_TIME_PATH = os.path.join(ROOT, 'exe_time')
os.makedirs(EXE_TIME_PATH, exist_ok=True)
CUMMULATIVE_COST_PATH = os.path.join(ROOT, 'cummulative_cost')
os.makedirs(CUMMULATIVE_COST_PATH, exist_ok=True)
TASK_BUCKETS_PATH = os.path.join(ROOT, 'task_buckets')
os.makedirs(TASK_BUCKETS_PATH, exist_ok=True)



BINS = [-1, 1474, 2211, 2579, 2763] # ALIBABA NEW
# m5d.large -> t2.xlarge/ m5.xlarge -> m7g.2xlarge -> m7g.4xlarge
VCPU_CATEGORY_PRICES = [0.113, 0.192, 0.326, 0.653] # Alibaba
    

def save_scheduling_state(fname, state_array):

    if not os.path.exists(STATE_PATH):
        os.mkdir(STATE_PATH)

    path = os.path.join(STATE_PATH, fname)
    with open(path+'.npy', 'wb') as f:
        np.save(f, np.array(state_array, dtype=object))
        print("SCHEDULING STATE SAVED!!")



def save_machine_count(fname, ct_mc_map):
    if not os.path.exists(MACHINE_CT_PATH):
        os.mkdir(MACHINE_CT_PATH)

    path = os.path.join(MACHINE_CT_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(ct_mc_map, f)
        print("MC CT MAP SAVED!!")

        

def save_duration_data(fname, duration_dict):

    if not os.path.exists(DURATION_PATH):
        os.mkdir(DURATION_PATH)

    path = os.path.join(DURATION_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(duration_dict, f)
        print("DURATIONS SAVED!!")


def save_batch_times(fname, batch_times_list):

    if not os.path.exists(BATCH_TIME_PATH):
        os.mkdir(BATCH_TIME_PATH)

    path = os.path.join(BATCH_TIME_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(batch_times_list, f)
        print("BATCH TIME SAVED!!")


def save_exe_time(fname, total_time):

    if not os.path.exists(EXE_TIME_PATH):
        os.mkdir(EXE_TIME_PATH)

    path = os.path.join(EXE_TIME_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(total_time, f)
        print("EXE TIME SAVED!!")

def save_cumm_cost(fname, cummulative_cost):

    if not os.path.exists(CUMMULATIVE_COST_PATH):
        os.mkdir(CUMMULATIVE_COST_PATH)

    path = os.path.join(CUMMULATIVE_COST_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(cummulative_cost, f)
        print("CUMMULATIVE COST SAVED!!")

def save_task_buckets(fname, task_buckets):
    if not os.path.exists(TASK_BUCKETS_PATH):
        os.mkdir(TASK_BUCKETS_PATH)

    path = os.path.join(TASK_BUCKETS_PATH, fname)
    with open(path+'.pkl', 'wb') as f:
        pickle.dump(task_buckets, f)
        print("TASK BUCKETS SAVED!!")


def sort_requests(containers, reverse=False):
    if reverse:
        containers = np.sort(containers.view('i8,i8,i8,i8,i8,i8,i8'), order=['f2','f3','f4','f5'], axis=0).view(int)[::-1]
    else:
        containers = np.sort(containers.view('i8,i8,i8,i8,i8,i8,i8'), order=['f2','f3','f4','f5'], axis=0).view(int)
    return containers

def sort_requests_all(containers, reverse=False):
    ts  = containers[:,1]
    if reverse:
        containers = np.sort(containers.view('i8,i8,i8,i8,i8,i8,i8'), order=['f2','f3','f4','f5'], axis=0).view(int)[::-1]
    else:
        containers = np.sort(containers.view('i8,i8,i8,i8,i8,i8,i8'), order=['f2','f3','f4','f5'], axis=0).view(int)
    containers[:,1] = ts

    return containers

def calculate_cost(dct, bins):
    total_machine_time = defaultdict(int)

    
    
    for key, values in dct.items():
        duration = 0
        for value in values:
            duration += (value[1] - value[0] + 1)
        total_machine_time[key] = duration
        
    my_list = list(total_machine_time.keys())
    
    # gives bin idx for each machine, 0-> 2vcpu, 1-> 4vcpu, ...
    bin_idx = np.digitize(my_list, bins, right=True) - 1
    cost = 0
    for i in range(len(my_list)):
        cost += (VCPU_CATEGORY_PRICES[bin_idx[i]] * total_machine_time[my_list[i]])
    
    return cost

def get_utlization(machines_availability, machines_capacity):

    usage = list()

    bin_idx = np.digitize(machines_capacity[:, 0], BINS, right=True) - 1

    avail = machines_availability[:,1:5]
    capacity = machines_capacity[:,1:5]


    # calculate vm utlization by category
    usage_t1, mc_t1, usage_t2, mc_t2, usage_t3, mc_t3, usage_t4, mc_t4 = 0, 0, 0, 0, 0, 0, 0, 0

    for idx in range(len(avail)):
        if np.any(avail[idx] != capacity[idx]):
            temp = ( (capacity[idx][0] - avail[idx][0]) / capacity[idx][0])

            if bin_idx[idx] == 0:
                usage_t1 += temp 
                mc_t1 += 1
            elif bin_idx[idx] == 1:
                usage_t2 += temp 
                mc_t2 += 1
            elif bin_idx[idx] == 2:
                usage_t3 += temp 
                mc_t3 += 1

            elif bin_idx[idx] == 3:
                usage_t4 += temp 
                mc_t4 += 4
        

    usage.append(usage_t1/mc_t1) if mc_t1 != 0 else usage.append(0.0)
    usage.append(usage_t2/mc_t2) if mc_t2 != 0 else usage.append(0.0)
    usage.append(usage_t3/mc_t3) if mc_t3 != 0 else usage.append(0.0)
    usage.append(usage_t4/mc_t4) if mc_t4 != 0 else usage.append(0.0)
    

    return usage

def print_active_machines(capacity, availability):
    avail_idx = list()
    for idx in range(len(capacity)):
        if np.any(capacity[idx, 1:5] != availability[idx, 1:5]):
            print(f"Machine {idx}: CAP: {capacity[idx, 1:5]}, AVAIL: {availability[idx, 1:5]}")
            avail_idx.append(idx)

    bin_idx = Counter(np.digitize(avail_idx, BINS, right=True) - 1)
    print(f"Type 1: {bin_idx[0]}, Type 2: {bin_idx[1]}, Type 3: {bin_idx[2]}, Type 4: {bin_idx[3]}")