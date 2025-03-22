import time, os, argparse
from src.episode import Episode
import numpy as np

# UTILS
from utils.machine_reader import generate_machine_list
from utils.container_reader import generate_container_list
from utils.monitoring import (
    BINS,
    save_scheduling_state, 
    save_machine_count, 
    save_duration_data, 
    save_batch_times, 
    save_exe_time, 
    calculate_cost,
    save_cumm_cost,
    save_task_buckets,
)

# ALGOS
from algos.heuristic.first_fit_batch import FirstFitBatch
from algos.heuristic.best_fit_batch import BestFitBatch
from algos.heuristic.tetris import Tetris
from algos.exact.vpsolver_batch import VpSolverBatch
from algos.rl.ppo import PPOBinPack
from algos.rl.gail import GAILBinPack
from algos.rl.bc import BCBinPack

parser = argparse.ArgumentParser(description='file')
parser.add_argument('--mf', help='machine file path')
parser.add_argument('--cf', help='container file path')
parser.add_argument('--l',  help='choose length', default=-1)
parser.add_argument('--grp_cnt',  help='grp cnt', default=-1)
parser.add_argument('--algo', help="Algorithm to use")
parser.add_argument('--state', help='State file name, save machine states, duration and machine count')

args = parser.parse_args()

machine_path = args.mf
contianer_path = args.cf
length = int(args.l)
algorithm = args.algo
state_fname = args.state
group_ct = int(args.grp_cnt)


machines, machines_dict = generate_machine_list(os.path.abspath(machine_path))
containers, containers_dict, static_ct = generate_container_list(os.path.abspath(contianer_path), length, group_ct)
print("#Containers=",len(containers), "#Containers Dict=",len(containers_dict), "#Machines=", len(machines))


DIVIDING_FACTOR = (3600) # 1k

total_time = 0
if algorithm == 'ffb':
    print()
    print("#####################################################")
    print("############### Algorithm: First Fit Batch ##########")
    print("#####################################################")
    time.sleep(1)
    algorithm = FirstFitBatch()
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()

    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

if algorithm == 'bfb':
    print()
    print("#####################################################")
    print("############### Algorithm: Best Fit Batch ###########")
    print("#####################################################")
    time.sleep(1)
    algorithm = BestFitBatch()
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()
    
    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

if algorithm == 'tetris':
    print()
    print("#####################################################")
    print("############### Algorithm: Tetris ###################")
    print("#####################################################")
    time.sleep(1)
    algorithm = Tetris()
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()
    
    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)


if algorithm == 'vpb':
    print()
    print("#####################################################")
    print("############### Algorithm: VpSolver Batch ###########")
    print("#####################################################")
    time.sleep(1)
    algorithm = VpSolverBatch(optimizer='GUROBI')
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()

    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

if algorithm == 'ppo':
    print()
    print("#####################################################")
    print("############### Algorithm: PPO ######################")
    print("#####################################################")
    time.sleep(1)


    # CONSTANTS
    # Change this path to the path of the best model
    BEST_MODEL_PPO_PATH_ROOT = '/data_ten/vectorized_bin_packing/RL/best_model/PPO/'



    ############################# ALIBABA COSTS #########################
    path = os.path.join(BEST_MODEL_PPO_PATH_ROOT, 'ppo_ckpt_alibaba_vm_3k_ms_1k_256_256_new_cost/rl_model_3251000_steps.zip')

    # VARIABLES
    ENV = 'vm-packing-ppo-google-real-cost-v0'

    env_config = {
        'step_limit' : np.size(containers, 0),
        'n_pms' : np.size(machines, 0),
        'machines' : machines[:,1:5],
        'max_vals' : machines[:,1:5].max(axis = 0),
        'containers_org' : containers,
        'train' : False,
    }

    algorithm = PPOBinPack(env_config, ENV, path)
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()

    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))
    

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

if algorithm == 'gail':
    print()
    print("#####################################################")
    print("############### Algorithm: GAIL #####################")
    print("#####################################################")
    time.sleep(1)


    # CONSTANTS
    BEST_MODEL_GAIL_PATH_ROOT = '/data_ten/vectorized_bin_packing/RL/best_model/GAIL/'

    # VARIABLES
    ENV = 'vm-packing-gail-real-duration-v0'

    env_config = {
        'step_limit' : np.size(containers, 0),
        'n_pms' : np.size(machines, 0),
        'machines' : machines[:,1:5],
        'max_vals' : machines[:,1:5].max(axis = 0),
        # 'containers' : containers,
        'containers_org' : containers,
        'cost' : machines[:,-1]/1000
    }
    
    ############################# ALIBABA 3K VMs 1K MS #########################
    path = os.path.join(BEST_MODEL_GAIL_PATH_ROOT,'gail_real_duration_ckpt_vp_i_mapping_grp_16_alibaba_1k_train_live_vm_3k_ms_1k_new/checkpoint00726/gen_policy/model.zip')

    algorithm = GAILBinPack(env_config, ENV, path)
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()

    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

if algorithm == 'bc':
    print()
    print("#####################################################")
    print("########## Algorithm: BEHAVIOUR CLONING #############")
    print("#####################################################")
    time.sleep(1)


    # CONSTANTS
    BEST_MODEL_BC_PATH_ROOT = '/data_ten/vectorized_bin_packing/RL/best_model/BC/'
    CKPT_BC_PATH_ROOT = '/data_ten/vectorized_bin_packing/RL/ckpts/bc_ckpts'

    # VARIABLES
    ENV = 'vm-packing-bc-real-duration-v0'

    env_config = {
        'step_limit' : np.size(containers, 0),
        'n_pms' : np.size(machines, 0),
        'machines' : machines[:,1:5],
        'max_vals' : machines[:,1:5].max(axis = 0),
        # 'containers' : containers,
        'containers_org' : containers,
        'cost' : machines[:,-1]/1000
    }
    


    ############################# ALIBABA #########################
    path = os.path.join(CKPT_BC_PATH_ROOT,'bc_real_duration_ckpt_vp_i_mapping_grp_16_alibaba_1k_train_live_vm_3k_ms_1k_new/checkpoint27543/gen_policy/model.zip')


    algorithm = BCBinPack(env_config, ENV, path)
    episode = Episode(machines, machines_dict, containers, containers_dict, algorithm, static_ct, bins=BINS)
    tic = time.time()
    episode.run()
    end = time.time()
    print()

    total_time = end - tic
    write_data = "Total Runtime: %f" %(total_time)
    print(write_data)

    ct_mc_map = episode.simulation.cluster.ct_mc_map
    print("MC=",len(set(ct_mc_map.values())))

    duration_dict = episode.simulation.cluster.duration
    cost = calculate_cost(duration_dict, BINS) / DIVIDING_FACTOR
    print("COST=",cost)

# STATE SAVING
if state_fname:

    scheduling_state_array = episode.simulation.cluster.events
    if scheduling_state_array:
        save_scheduling_state(state_fname, scheduling_state_array)
    
    ct_mc_map = episode.simulation.cluster.ct_mc_map
    if ct_mc_map:
        save_machine_count(state_fname, ct_mc_map)

    duration_dict = episode.simulation.cluster.duration
    if duration_dict:
        save_duration_data(state_fname, duration_dict)

    batch_times_list = algorithm.batch_times
    if batch_times_list:
        save_batch_times(state_fname, batch_times_list)

    if total_time != 0:
        save_exe_time(state_fname, total_time)

    cummulative_cost = episode.simulation.cluster.cummulative_cost
    if cummulative_cost:
        save_cumm_cost(state_fname, cummulative_cost)

    task_buckets = episode.simulation.cluster.bucket_sizes
    if task_buckets:
        save_task_buckets(state_fname, task_buckets)

    
