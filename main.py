from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gym
import pandas as pd

from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.state import PostgresState
from dbgym.spaces.observations.qppnet.features import QPPNetFeatures
from dbgym.workload.manager import WorkloadManager
from dbgym.workload.transform import (
    OnlyUniqueQueries,
    ParamSubstitution,
    PerfectPrediction,
    SampleFracQueries,
)
from models.qppnet import QPPNet
from tqdm import tqdm


@dataclass
class ExperimentConfig:
    name: str
    save_path: Path
    train_csvlog_path: Path
    test_csvlog_path: Optional[Path]


state_path = Path("./artifact/prod_dbms/state")
experiment_configs = [
    ExperimentConfig(
        name="default",
        save_path=Path("./artifact/experiment/default/"),
        train_csvlog_path=Path("./artifact/prod_dbms/workload_train.csv"),
        test_csvlog_path=Path("./artifact/prod_dbms/workload_test.csv"),
    ),
]

pbar = tqdm(total=len(experiment_configs), desc="Iterating over experiment configs.")
for config in experiment_configs:
    pbar.set_description(desc=f"Config: {config.name}")
    workload_manager = WorkloadManager(
        save_name=config.name,
        save_path=config.save_path,
        train_csvlog_path=config.train_csvlog_path,
        test_csvlog_path=config.test_csvlog_path,
    )
    workload_manager.add_transform(PerfectPrediction())
    # workload_manager.add_transform(OnlyUniqueQueries())
    # workload_manager.add_transform(SampleFracQueries(frac=0.01, random_state=15721))
    # workload_manager.add_transform(SampleFracQueries(frac=0.1, random_state=15721))
    # workload_manager.add_transform(SampleFracQueries(frac=0.8, random_state=15721))
    # workload_manager.add_transform(ParamSubstitution("const0", lambda series: f"'0'"))
    # workload_manager.add_transform(ParamSubstitution("mean", lambda series: f"'{series.mean()}'"))
    # workload_manager.add_transform(ParamSubstitution("median", lambda series: f"'{series.median()}'"))
    # workload_manager.add_transform(ParamSubstitution("max", lambda series: f"'{series.max()}'"))
    # workload_manager.add_transform(ParamSubstitution("min", lambda series: f"'{series.min()}'"))

    train_df, test_df, validation_df = None, None, None
    experiment_workloads = workload_manager.generate_experiment_workloads()
    ppbar = tqdm(total=len(experiment_workloads), desc="Iterating over experiment workloads.", leave=False)
    for experiment_workload in experiment_workloads:
        ppbar.set_description(desc=f"Workload: {experiment_workload.name}")
        iteration = 0
        gym_spec = GymSpec(
            historical_workloads=experiment_workload.workloads,
            historical_state=PostgresState(state_path),
        )

        observations_path = config.save_path / experiment_workload.name / f"observations_{iteration}.parquet"
        observations_path.parent.mkdir(parents=True, exist_ok=True)
        if not observations_path.exists():
            # The env_checker is pretty slow, taking about 200s for 1.2m datapoints.
            # Obviously, debug with the env_checker enabled! It catches bugs.
            env = gym.make(
                "dbgym/DbGym-v0",
                disable_env_checker=True,
                gym_spec=gym_spec,
                try_prepare=False,
                print_errors=True
            )
            # TODO(WAN): Need to further abstract for (1) different model types and (2) actual action selection.
            #            But until I figure out what the APIs for those look like, this will do.
            assert isinstance(env.observation_space, QPPNetFeatures)
            observations, info = env.reset(seed=15721)
            train_df = env.observation_space.convert_observations_to_df(observations)
            train_df.to_parquet(observations_path)
            env.close()

        # Special case for the OnlyTest experiment.
        # Don't predict anything, just save the dataframe for later.
        # This is done because all the experiments use the same test DF.
        if experiment_workload.name == "OnlyTest":
            test_df = pd.read_parquet(observations_path)
            test_df["Query Hash"] = test_df["Query Hash"].apply(tuple)
            _, validation_df = QPPNet.split(test_df, min_test_size=len(test_df), random_state=15721)
            continue

        train_df = pd.read_parquet(observations_path)
        train_df["Query Hash"] = train_df["Query Hash"].apply(tuple)

        destination_folder = config.save_path / experiment_workload.name / "qppnet"
        destination_folder.mkdir(parents=True, exist_ok=True)
        assert train_df is not None and validation_df is not None and test_df is not None
        model = QPPNet(
            train_df,
            test_df,
            save_folder=destination_folder,
            batch_size=256,
            num_epochs=1000,
            patience=5,
            patience_min_epochs=None,
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
        ppbar.update(1)
    ppbar.close()
    pbar.update(1)
