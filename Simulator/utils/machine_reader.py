# This file reads machines csv file and creates machine objects
import pandas as pd
import numpy as np

def generate_machine_list(file_name):
    df = pd.read_csv(file_name)

    machines_list = list()
    machines_dict = {}
    
    disk_flag, net_flag, cost_flag = False, False, False


    if 'disk' in df.columns:
        disk_flag = True

    if 'net' in df.columns:
        net_flag = True

    if 'cost' in df.columns:
        cost_flag = True 

    for idx, row in df.iterrows():

        disk , net, cost = -1, -1, 0
        if disk_flag:
            disk = row.disk 
        if net_flag:
            net = row.net        
        if cost_flag:
            cost = int(row.cost * 1000)

        machines_dict[idx] = row.mid 

        mid = int(row.mid.split('_')[1])
        machine = np.array([idx, row.cpu, row.mem, disk, net, cost])
        machines_list.append(machine)

    machines = np.array(machines_list, dtype=int)
    return machines, machines_dict