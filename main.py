import gym
import dbgym

from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from dbgym.envs.workload import Workload
from dbgym.envs.state import PostgresState
from dbgym.envs.gym_spec import GymSpec

from pathlib import Path

import pandas as pd

from dbgym.envs.trainer import PostgresTrainer

trainer = PostgresTrainer(None, None)
trainer.create_target_dbms()

# csvlog_path = Path("./artifact/prod_dbms/workload.csv")
# workload_path = Path("./artifact/prod_dbms/workload.db")
# state_path = Path("./artifact/prod_dbms/state")
#
# # convert_postgresql_csvlog_to_workload(csvlog_path, workload_path)
# workload = Workload(workload_path)
# state = PostgresState(state_path)
#
# gym_spec = GymSpec("postgresql+psycopg2://gym_user:gym_pass@127.0.0.1/gym_db",
#                    historical_workload=workload,
#                    historical_state=state,
#                    wan=True)
# env = gym.make("dbgym/DbGym-v0", gym_spec=gym_spec)
#
# observation, info = env.reset(seed=15721)

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset(return_info=True)
# env.close()
