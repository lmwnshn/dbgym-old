from gymnasium.envs.registration import register

register(
    id="dbgym/DbGym-v0",
    entry_point="dbgym.env.dbgym:DbGymEnv",
    max_episode_steps=300,
)
