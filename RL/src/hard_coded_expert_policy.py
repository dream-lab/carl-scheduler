from imitation.policies.base import HardCodedPolicy
import numpy as np

class VPSolverPolicy(HardCodedPolicy):

    def __init__(self, env, run_until, expert):
        self.env = env 
        self.curr_step = self.env.current_step
        self.run_until = run_until
        self.expert = expert

    def predict(self, observation, deterministic = False):

        # midx = self.env.containers[self.env.current_step][1]
        midx = self.expert[self.env.current_step]
        # print(f"VP current_step={self.curr_step}, env_step={self.env.current_step}, midx={midx}")
        acts = np.array([midx])
        self.curr_step = (self.curr_step + 1)%self.run_until
        return acts, None

    def _choose_action(self, obs):
        # return np.array([self.env.containers[self.env.current_step][1]])
        return np.array([self.env.expert[self.env.current_step]])
