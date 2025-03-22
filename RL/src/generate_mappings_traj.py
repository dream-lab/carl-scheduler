import sys
sys.path.append('../../')
import os, argparse, pickle, time, pickle

# Imports
from Simulator.utils.machine_reader import generate_machine_list
from Simulator.utils.container_reader import generate_container_list
from generate_expert_traj_vp import vp_mappings

# CLI Arguments
parser = argparse.ArgumentParser(description='file')
parser.add_argument('--mf', help='machine file path')
parser.add_argument('--cf', help='container file path')
parser.add_argument('--nm', help='mapping name')
parser.add_argument('--b',  help='choose batch', default=16)
parser.add_argument('--grp_cnt',  help='grp cnt', default=-1)


args = parser.parse_args()

machine_path = args.mf
contianer_path = args.cf
batch = int(args.b)
name = args.nm
group_ct = int(args.grp_cnt)



# Generating data
machines, machines_dict = generate_machine_list(os.path.abspath(machine_path))
containers, containers_dict, static_ct = generate_container_list(os.path.abspath(contianer_path), group_ct=group_ct)

model = None 
actions_dict = None

max_vals = machines[:,1:].max(axis = 0)

tic = time.time()

print("Generating Mappings!")
# batch = len(containers)
actions_dict = vp_mappings(machines, containers, batch)
toc = time.time()
print("Done=",len(actions_dict))


with open('../vp_mappings/'+name+'.pkl', 'wb') as f:
    pickle.dump(actions_dict, f)

print("Mappings generated!!")
print("Time=",toc-tic)
