from gym.envs.registration import register

register(
    id="dbgym/DbGym-v0",
    entry_point="dbgym.envs.dbgym_env:DbGymEnv",
    max_episode_steps=300,
)
