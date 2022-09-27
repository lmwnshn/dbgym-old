import gym
import numpy as np

import dbgym

from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from dbgym.envs.workload import Workload
from dbgym.envs.state import PostgresState
from dbgym.envs.gym_spec import GymSpec

from pathlib import Path

import pandas as pd

import time

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
# The env_checker is pretty slow, taking about 200s for 1.2m datapoints.
# Obviously, debug with the env_checker enabled! It catches bugs.
env = gym.make("dbgym/DbGym-v0", gym_spec=gym_spec, disable_env_checker=True)
observations, info = env.reset(seed=15721)

print("We here now", time.time())
start = time.time()
df = pd.json_normalize(observations)
end = time.time()
print(f"{end - start:.2f} s for json_normalize")

start = time.time()

explode = False
# Until this flag is fixed, categories cannot be dicts.
# pyarrow.lib.ArrowNotImplementedError: Writing DictionaryArray with nested dictionary type not yet supported
# Instead, use the following block of code on reading the parquet file.
# for cat in [
#     "Index Name",
#     "Join Type",
#     "Node Position",
#     "Node Type",
#     "Parent Relationship",
#     "Partial Mode",
#     "Query Hash",
#     "Relation Name",
#     "Sort Method",
#     "Scan Direction",
#     "Strategy",
# ]:
#     df[cat] = df[cat].apply(tuple).astype("category")

for col in df.columns:
    if type(df[col].iloc[0]) in [tuple, list, np.ndarray]:
        lens = df[col].apply(len)
        # If this is a Box shape, unwrap the ndarray value.
        if (lens == 1).all():
            df[col] = df[col].apply(lambda arr: arr.item())
        elif explode:
            # Otherwise, find the maximum length and explode out.
            max_len = lens.max()
            df[col] = df[col].apply(lambda x: np.array(x).flatten())
            if not (lens == max_len).all():
                # Pad.
                df[col] = df[col].apply(lambda arr: np.pad(arr, (0, max_len - arr.shape[0])))
            new_cols = [f"{col}_{i}" for i in range(1, max_len + 1)]
            df[new_cols] = pd.DataFrame(df[col].tolist(), index=df.index)
            del df[col]

assert df["Observation Index"].nunique() == len(df), "Observation not unique?"
df = df.sort_values("Observation Index").reset_index(drop=True)
assert all(df["Observation Index"] == df.index), "Missing observations?"

df["Node Position"] = df["Node Position"].apply(np.hstack)
# df["Node Position"] = df["Node Position"].apply(lambda x: tuple(np.hstack(x))).astype("category")
df["Children Indexes"] = df["Children Indexes"].apply(lambda x: [idxs for sublist in x for idxs in sublist])

categories = [
    "Query Num",
]
for category in categories:
    df[category] = df[category].astype("category")

# onehots = [
#     "Index Name",
#     "Join Type",
#     "Node Type",
#     "Parent Relationship",
#     "Partial Mode",
#     "Query Hash",
#     "Relation Name",
#     "Sort Method",
#     "Scan Direction",
#     "Strategy",
# ]
# for onehot in onehots:
#     df[onehot] = df[onehot].apply(tuple).astype("category")

# Defragment the df.
df = df.copy()
df.to_parquet("./artifact/gym/data.parquet")

end = time.time()
print(f"{end - start:.2f} s for processing")

# Train a model from these observations.

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset(return_info=True)
# env.close()
