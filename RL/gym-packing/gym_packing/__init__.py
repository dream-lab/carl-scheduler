from gym.envs.registration import register


register(
    id='vm-packing-gail-real-duration-v0',
    entry_point='gym_packing.envs:VMPackingEnvGAILRealDuration',
)


register(
    id='vm-packing-ppo-google-real-cost-v0',
    entry_point='gym_packing.envs:VMPackingEnvPPOGoogleRealCost',
)

register(
    id='vm-packing-bc-real-duration-v0',
    entry_point='gym_packing.envs:VMPackingEnvBCRealDuration',
)