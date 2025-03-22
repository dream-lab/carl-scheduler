# Training script for Behavioural Cloning
import sys 
sys.path.append('../../')

# Imports
import os, argparse, pickle, copy
import torch as th
import numpy as np

import pathlib
import gym

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO



from imitation.algorithms.bc import BC
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout
from imitation.util.util import make_vec_env

from imitation.scripts.train_adversarial import save


# MODULE IMPORTS
import gym_packing
from hard_coded_expert_policy import VPSolverPolicy
from Simulator.utils.machine_reader import generate_machine_list
from Simulator.utils.container_reader import generate_container_list


# CONSTANTS
ENV = 'vm-packing-bc-real-duration-v0'

MAPPINGS_PATH_ROOT =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vp_mappings') 
BEST_MODEL_BC_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'best_model/BC/') 
CKPT_BC_PATH_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ckpts/bc_ckpts')

os.makedirs(MAPPINGS_PATH_ROOT, exist_ok=True)
os.makedirs(BEST_MODEL_BC_PATH_ROOT, exist_ok=True)
os.makedirs(CKPT_BC_PATH_ROOT, exist_ok=True)

# GOOGLE NEW EXP
CKPT_FOLDER_NAME = ''
BC_MODEL_NAME = ''
# VP Solver Mapping
MAPPING_FILE_NAME = # '.pkl' # Generate pkl mapping file name
BATCH_SIZE = 1024


# CLI Arguments
parser = argparse.ArgumentParser(description='file')
parser.add_argument('--mf', help='machine file path')
parser.add_argument('--cf', help='container file path')
parser.add_argument('--l',  help='choose length', default=-1)
parser.add_argument('--check', help="Validate gym environment", default=0)
parser.add_argument('--grp_cnt',  help='grp cnt', default=-1)
args = parser.parse_args()

machine_path = args.mf
contianer_path = args.cf
check = int(args.check)
length = int(args.l)
group_ct = int(args.grp_cnt)


# Generating data
machines, machines_dict = generate_machine_list(os.path.abspath(machine_path))
containers, containers_dict, static_ct = generate_container_list(os.path.abspath(contianer_path), length, group_ct)

env_config = {
    'step_limit' : np.size(containers, 0),
    'n_pms' : np.size(machines, 0),
    'machines' : machines[:,1:5],
    'max_vals' : machines[:,1:5].max(axis = 0),
    'containers_org' : containers,
    'cost' : machines[:,-1]/1000
}


if check == 1:
    from stable_baselines3.common.env_checker import check_env
    env = gym.make(ENV, env_config=env_config)
    check_env(env)


actions_dict = None
with open(os.path.join(MAPPINGS_PATH_ROOT, MAPPING_FILE_NAME), 'rb') as f:
    actions_dict = pickle.load(f)

cids = sorted([c for c in actions_dict.keys()])
mids = [actions_dict[c] for c in cids]


transitions = None


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.valid_action_mask()

rng = np.random.default_rng(0)
env = gym.make(ENV, env_config=env_config)
env = ActionMasker(env, mask_fn)
env = RolloutInfoWrapper(env)
venv = DummyVecEnv([lambda : env])

run_till = env_config['step_limit']
expert = VPSolverPolicy(env, run_till, mids)

print("Rollouts Started!!")
rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=run_till, min_episodes=None),
    rng=rng,
)
print(f"obs shape={rollouts[0].obs.shape}")
print("Rollouts Ended!!")

transitions = rollout.flatten_trajectories(rollouts)

venv = make_vec_env(ENV, n_envs=1, env_make_kwargs=env_config, rng=rng)
env = gym.make(ENV, env_config=env_config)
env = ActionMasker(env, mask_fn)  # Wrap to enable masking



venv = DummyVecEnv([lambda: env])
# Custom actor (pi) and value function (vf) networks
# of two layers of size 64, 32 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=dict(pi=[256, 256], vf=[256, 256]))

policy = MaskableActorCriticPolicy(
    lr_schedule=lambda _: 3e-4,
    observation_space=env.observation_space,
    action_space=env.action_space,
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=th.nn.ReLU,
)


bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    batch_size=BATCH_SIZE,
    rng=rng,
    policy=policy,
)


total_timesteps = 10000000
checkpoint_interval = env_config['step_limit']
ckpt_path = os.path.join(CKPT_BC_PATH_ROOT, CKPT_FOLDER_NAME)


from imitation.policies import serialize
def save(trainer, save_path: pathlib.Path):
    """Save discriminator and generator."""
    save_path.mkdir(parents=True, exist_ok=True)
    serialize.save_stable_model(
        save_path / "gen_policy",
        trainer.policy,
    )

round_num = 0
def callback() -> None:
    global round_num
    if round_num >= 5000000 and round_num % checkpoint_interval == 0:
        save(bc_trainer, pathlib.Path(f"{ckpt_path}/checkpoint{round_num}"))
    round_num += 1

print("Training Started!!")
bc_trainer.train(n_epochs=total_timesteps, on_epoch_end=callback, log_interval=4)
print("Training done!")


# SAVING
save(bc_trainer, pathlib.Path(f"{BEST_MODEL_BC_PATH_ROOT}/{BC_MODEL_NAME}"))
    