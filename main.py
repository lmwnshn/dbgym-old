from pathlib import Path

import gym
import pandas as pd

from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.state import PostgresState
from dbgym.envs.workload import Workload
from dbgym.spaces.qppnet_features import QPPNetFeatures
from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from dbgym.workload.split import split_workload
from dbgym.workload.transform import (Identity, OnlyUniqueQueries,
                                      ParamSubstitution, SampleFracQueries)
from models.qppnet import QPPNet

csvlog_path = Path("./artifact/prod_dbms/workload.csv")
state_path = Path("./artifact/prod_dbms/state")

# Create folders.
Path("./artifact/gym").mkdir(parents=True, exist_ok=True)
Path("./artifact/qppnet/observations").mkdir(parents=True, exist_ok=True)
Path("./artifact/qppnet/model").mkdir(parents=True, exist_ok=True)
Path("./artifact/experiment/transformed_workloads/").mkdir(parents=True, exist_ok=True)

workload_path = Path("./artifact/gym/workload.db")
workload_train_path = Path("./artifact/gym/workload.train.db")
workload_test_path = Path("./artifact/gym/workload.test.db")

observations_path = Path("./artifact/qppnet/observations")
model_path = Path("./artifact/qppnet/model")
transformed_workloads_path = Path("./artifact/experiment/transformed_workloads/")

# Transforms to try.
transforms = [
    Identity(),
    OnlyUniqueQueries(),
    SampleFracQueries(frac=0.01, random_state=15721),
    SampleFracQueries(frac=0.1, random_state=15721),
    SampleFracQueries(frac=0.8, random_state=15721),
    ParamSubstitution("const0", lambda series: f"'0'"),
    # ParamSubstitution("max", lambda series: f"'{series.max()}'"),
    ParamSubstitution("mean", lambda series: f"'{series.mean()}'"),
    ParamSubstitution("median", lambda series: f"'{series.median()}'"),
    # ParamSubstitution("min", lambda series: f"'{series.min()}'"),
]

# Experiments to be run. Test must be first.
experiments = [
    ("Test", [Workload(workload_test_path)]),
    # And one for each transform.
]

# Create the workload if it doesn't exist.
if not workload_path.exists():
    convert_postgresql_csvlog_to_workload(csvlog_path, workload_path)
if not (workload_train_path.exists() and workload_test_path.exists()):
    split_workload(workload_path, workload_train_path, workload_test_path)

# Create the transformed DBs if necessary, add them to the experiments list.
for transform in transforms:
    workload_transform_path = (
        transformed_workloads_path / f"workload_{transform.name}.db"
    )
    # TODO(WAN): dangerous semantics here.
    if not workload_transform_path.exists():
        transform.transform(workload_test_path, workload_transform_path)
    experiments.append(
        (
            transform.name,
            [Workload(workload_train_path), Workload(workload_transform_path)],
        )
    )

# TODO(WAN): State.
state = PostgresState(state_path)

# Run each experiment.
train_df, validation_df, test_df = None, None, None
for experiment_name, workloads in experiments:
    print(f"Experiment: {experiment_name}")

    iteration = 0
    gym_spec = GymSpec(
        historical_workloads=workloads,
        historical_state=state,
        wan=False,
    )

    destination_pq = (
        observations_path / f"{experiment_name}_iteration_{iteration}.parquet"
    )
    if not destination_pq.exists():
        # The env_checker is pretty slow, taking about 200s for 1.2m datapoints.
        # Obviously, debug with the env_checker enabled! It catches bugs.
        env = gym.make("dbgym/DbGym-v0", gym_spec=gym_spec, disable_env_checker=True)
        observations, info = env.reset(seed=15721)
        if isinstance(env.observation_space, QPPNetFeatures):
            train_df = env.observation_space.convert_observations_to_df(observations)
            train_df.to_parquet(destination_pq)
        env.close()

    # Special case for the Test experiment.
    # Don't predict anything, just save the dataframe for later.
    # This is done because all the experiments use the same test DF.
    if experiment_name == "Test":
        test_df = pd.read_parquet(destination_pq)
        for cat in [
            "Query Hash",
        ]:
            test_df[cat] = test_df[cat].apply(tuple)
        _, validation_df = QPPNet.split(test_df, test_size=512, random_state=15721)
        continue

    train_df = pd.read_parquet(destination_pq)
    for cat in [
        "Query Hash",
    ]:
        train_df[cat] = train_df[cat].apply(tuple)

    destination_folder = model_path / experiment_name
    destination_folder.mkdir(parents=True, exist_ok=True)

    assert train_df is not None and validation_df is not None and test_df is not None
    model = QPPNet(
        train_df,
        test_df,
        save_folder=destination_folder,
        batch_size=256,
        num_epochs=4000,
        validation_size=4096,
    )
    model.train(epoch_save_interval=100, validation_df=validation_df)

    # # Pick actions based on the model.
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     observations, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         observations, info = env.reset()
