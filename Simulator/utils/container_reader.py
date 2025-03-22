# This file reads container csv file and creates machine objects
import pandas as pd
import numpy as np

def generate_container_list(file_name, length=-1, group_ct=-1):
    df = pd.read_csv(file_name)

    disk_flag, net_flag = False, False 

    if 'disk' in df.columns:
        disk_flag = True

    if 'net' in df.columns:
        net_flag = True


    container_list = list()
    containers_dict = {}
    static_ct = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx == length:
            break

        disk , net = -1, -1
        if disk_flag:
            disk = row.disk 
        if net_flag:
            net = row.net

        containers_dict[idx] = row.cid
        

        if row.duration == -1:
            container_list.append([idx, row.ts, row.cpu, row.mem, disk, net, row.duration])
        else:
            static_ct += 1
            container_list.append([idx, row.ts, row.cpu, row.mem, disk, net, row.duration])

    
    containers = np.array(container_list, dtype=int)

    if group_ct != -1:
        i = 0
        list_len = len(containers)
        ans = []
        while len(ans) < list_len:
            ans.extend([i]*group_ct)
            i += 1

        ans = ans[:list_len]
        containers[:,1] = np.array(ans, dtype=int)


    return containers, containers_dict, static_ct
