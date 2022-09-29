from pathlib import Path

import gym

from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.state import PostgresState
from dbgym.envs.workload import Workload
from dbgym.spaces.qppnet_features import QPPNetFeatures
from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from models.qppnet import QPPNet

csvlog_path = Path("./artifact/prod_dbms/workload.csv")
state_path = Path("./artifact/prod_dbms/state")

Path("./artifact/gym").mkdir(parents=True, exist_ok=True)
Path("./artifact/qppnet/observations").mkdir(parents=True, exist_ok=True)
Path("./artifact/qppnet/model").mkdir(parents=True, exist_ok=True)
workload_path = Path("./artifact/gym/workload.db")
observations_path = Path("./artifact/qppnet/observations")
model_path = Path("./artifact/qppnet/model")

if not workload_path.exists():
    convert_postgresql_csvlog_to_workload(csvlog_path, workload_path)

workload = Workload(workload_path)
state = PostgresState(state_path)

iteration = 0
gym_spec = GymSpec(
    "postgresql+psycopg2://prod_user:prod_pass@127.0.0.1:5432/prod_db",
    historical_workload=workload,
    historical_state=state,
    wan=False,
)
# The env_checker is pretty slow, taking about 200s for 1.2m datapoints.
# Obviously, debug with the env_checker enabled! It catches bugs.
env = gym.make("dbgym/DbGym-v0", gym_spec=gym_spec, disable_env_checker=True)
observations, info = env.reset(seed=15721)

if isinstance(env.observation_space, QPPNetFeatures):
    df = env.observation_space.convert_observations_to_df(observations)
    df.to_parquet(observations_path / f"iteration_{iteration}.parquet")
    query_nums = sorted(df["Query Num"].unique())
    midpoint = query_nums[len(query_nums) // 2]
    train_df, test_df = df[df["Query Num"] <= midpoint], df[df["Query Num"] > midpoint]
    model = QPPNet(train_df, test_df, save_folder=model_path, batch_size=512, num_epochs=25000)
    model.train(epoch_save_interval=1000, evaluate_on_save=True)
# breakpoint()
#
# # Train a model from these observations.
# for _ in range(1000):
#     action = env.action_space.sample()
#     observations, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observations, info = env.reset()
env.close()
