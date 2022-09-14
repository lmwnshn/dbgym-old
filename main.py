import gym
import dbgym

from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from dbgym.envs.workload import Workload
from dbgym.envs.state import PostgresState
from dbgym.envs.gym_spec import GymSpec

from pathlib import Path

import pandas as pd

csvlog_path = Path("./artifact/prod_dbms/workload.csv")
state_path = Path("./artifact/prod_dbms/state")

Path("./artifact/gym").mkdir(parents=True, exist_ok=True)
workload_path = Path("./artifact/gym/workload.db")

if not workload_path.exists():
    convert_postgresql_csvlog_to_workload(csvlog_path, workload_path)

workload = Workload(workload_path)
state = PostgresState(state_path)

gym_spec = GymSpec("postgresql+psycopg2://prod_user:prod_pass@127.0.0.1:5432/prod_db",
                   historical_workload=workload,
                   historical_state=state,
                   wan=True)
env = gym.make("dbgym/DbGym-v0", gym_spec=gym_spec)

observation, info = env.reset(seed=15721)

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset(return_info=True)
# env.close()
