# CARL SCHEDULER

## Citation
Prathamesh Saraf Vinayak; Saswat Subhajyoti Mallick; Lakshmi Jagarlamudi; Anirban Chakraborty; Yogesh Simmhan; **CARL: Cost-Optimized Online Container Placement on VMs Using Adversarial Reinforcement Learning** in *IEEE Transactions on Cloud Computing*, vol. 13, no. 1, pp. 321-335, Jan.-March 2025, doi: [10.1109/TCC.2025.3528446](https://doi.org/10.1109/TCC.2025.3528446).

The CARL Scheduler code was developed by Prathamesh and Saswat from the Cloud Systems Lab (CSL) at IISc.

## Installation

```bash
pip install -r requirements.txt
```

**Note:**

- For vpsolver, you need to install gurobi and add it to your path.
- For RL training, you need to install gym. Currently, gym has been upgraded to gymnasium, better to use that for speed and efficiency. Ideally port the code to gymnasium.
- The way RL envs are registered and uses is different in the new gymnasium.
- All the libraries such as sympy and all are upgraded, the current code is using the older versions. Best to port them to access the new features and stability.
- I had to hack the library code for gym and RL to work.
- Please try to port the code to latest sympy, gymnasium, imitation, stable-baselines3, etc.

Links:

- Gym: https://www.gymlibrary.dev/index.html
- Gymnasium: https://gymnasium.farama.org/
- A2C/PPO: https://stable-baselines3.readthedocs.io/en/master/index.html
- GAIL: https://imitation.readthedocs.io/en/latest/

## Usage

- Refer Makefile for evaluation
- Simulator code is in `Simulator/`
- RL training code is in `RL/`
- You need to generate expert trajectories first, refer to `generate_expert_traj_vp.py` in `RL/`, need access to Gurobi
- To train RL models, refer to `commands.txt` in `RL/`, installed libary versions require changes to the library code. Better to install newer libraries and make them work.
