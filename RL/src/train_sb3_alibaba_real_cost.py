import sys 
sys.path.append('../../')

# Imports
import os, argparse, pickle
import torch as th
import gym

import numpy as np

from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO


from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback, CallbackList


# MODULES
import gym_packing
from Simulator.utils.container_reader import generate_container_list
from Simulator.utils.machine_reader import generate_machine_list

# CONSTANTS
MAPPINGS_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vp_mappings') 
BEST_MODEL_PPO_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model/PPO/') 
PPO_TBOARD_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tensorboard_logs/PPO/') 
CKPT_PPO_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ckpts/ppo_ckpts')

os.makedirs(MAPPINGS_PATH_ROOT, exist_ok=True)
os.makedirs(BEST_MODEL_PPO_PATH_ROOT, exist_ok=True)
os.makedirs(PPO_TBOARD_LOG_PATH, exist_ok=True)
os.makedirs(CKPT_PPO_PATH_ROOT, exist_ok=True)

# VARIABLES
ENV = 'vm-packing-ppo-alibaba-real-cost-v0'
# Generate pkl mapping file name
MAPPING_FILE_NAME = #'.pkl' 
CKPT_FOLDER_NAME = '' # ppo ckpt folder name


# CLI Arguments
parser = argparse.ArgumentParser(description='file')
parser.add_argument('--mf', help='machine file path')
parser.add_argument('--cf', help='container file path')
parser.add_argument('--check', help="Validate gym environment", default=0)
parser.add_argument('--train', help="Train gym environment", default=0)
parser.add_argument('--gen_expert_traj', help="generate expert traj", default=0)

args = parser.parse_args()

machine_path = args.mf
contianer_path = args.cf
check = int(args.check)
train = int(args.train)
expert_traj = int(args.gen_expert_traj)




# Generating data
machines, machines_dict = generate_machine_list(os.path.abspath(machine_path))
containers, containers_dict, static_ct = generate_container_list(os.path.abspath(contianer_path))

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()



# print(machines[:,1:].max(axis = 0), containers.max(axis=0))
print("Mids=",len(machines))
print("Cids=",len(containers))

model = None 
actions_dict = None
env_config = {
    'step_limit' : np.size(containers, 0),
    'n_pms' : np.size(machines, 0),
    'machines' : machines[:,1:5],
    'max_vals' : machines[:,1:5].max(axis = 0),
    'containers_org' : containers,
    'shuffle_cnt' : True,
    'cost' : machines[:,-1]/1000
}

# print(env_config['cost'][:5])

if check == 1:
    
    from stable_baselines3.common.env_checker import check_env
    env = gym.make(ENV, env_config=env_config)
    env = ActionMasker(env, mask_fn)
    check_env(env)


    
trainer = None

if train == 1:

    actions_dict = None
    transitions = None
    venv = None
    env = None

    if expert_traj == 1:
        with open(os.path.join(MAPPINGS_PATH_ROOT, MAPPING_FILE_NAME), 'rb') as f:
            actions_dict = pickle.load(f)

        # print(actions_dict)
        cids = sorted([c for c in actions_dict.keys()])
        mids = [mid for mid in actions_dict.values()]

        print("Mids=",len(set(mids)))
        print("Cids=",len(cids))

        containers = containers[cids, :]
        env_config = {
            'step_limit' : np.size(containers, 0),
            'n_pms' : np.size(machines, 0),
            'machines' : machines[:,1:5],
            'max_vals' : machines[:,1:5].max(axis = 0),
            'containers_org' : containers,
            'shuffle_cnt' : True,
            'cost' : machines[:,-1]/1000
        }

    eval_env = gym.make(ENV, env_config=env_config)
    eval_env = ActionMasker(eval_env, mask_fn)
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=BEST_MODEL_PPO_PATH_ROOT,
        log_path=PPO_TBOARD_LOG_PATH,
        eval_freq=env_config['step_limit']*5,
        deterministic=True, 
        render=False,
        verbose=0,
    )

    
    ckpt_path = os.path.join(CKPT_PPO_PATH_ROOT, CKPT_FOLDER_NAME)
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config['step_limit'],
        save_path=ckpt_path,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )


    progress_bar = ProgressBarCallback()
    callbacks = CallbackList([eval_callback, progress_bar, checkpoint_callback])

    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    env = gym.make(ENV, env_config=env_config)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # Custom actor (pi) and value function (vf) networks
    # of two layers of size 32 each with Relu activation function
    policy_kwargs = dict(
        # net_arch=dict(vf=[256, 256], pi=[256, 256]),
        net_arch=dict(vf=[64, 128], pi=[64, 128]),
        activation_fn=th.nn.ReLU,
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy, 
        env, 
        verbose=0, 
        tensorboard_log=PPO_TBOARD_LOG_PATH, 
        batch_size=64, 
        policy_kwargs=policy_kwargs,
        gamma=0.8,
        seed=1,
    )

    model.learn(15000000, callback=callbacks)
    print("Training done!!")
