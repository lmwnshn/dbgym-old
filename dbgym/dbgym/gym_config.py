from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium
from dbgym.config import Config
from dbgym.db_config import DbConfig
from dbgym.env.dbgym import DbGymEnv
from dbgym.nyoom import nyoom_start, nyoom_stop
from dbgym.space.action.fake_index import FakeIndexSpace
from dbgym.space.observation.qppnet.features import QPPNetFeatures
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.workload.workload import Workload


@dataclass
class GymConfig:
    db_config: DbConfig
    expt_name: str
    workload: list[Workload]
    seed: int
    should_nyoom: bool
    should_overwrite: bool
    setup_sql: list[str]
    save_path_observation: Path = Config.SAVE_PATH_OBSERVATION
    nyoom_args: Optional[dict] = None

    def __post_init__(self):
        self.obs_path = self.save_path_observation / self.expt_name
        self.obs_path.mkdir(parents=True, exist_ok=True)
        self.pq_path = self.obs_path / f"0.parquet"

    def should_run(self):
        return not self.pq_path.exists() or self.should_overwrite

    def force_run(self, db_snapshot: DatabaseSnapshot):
        if self.should_nyoom:
            nyoom_start(self.db_config, self.nyoom_args)

        action_space = FakeIndexSpace(1)
        observation_space = QPPNetFeatures(db_snapshot=db_snapshot, seed=self.seed)

        # noinspection PyTypeChecker
        env: DbGymEnv = gymnasium.make(
            "dbgym/DbGym-v0",
            disable_env_checker=True,
            name=self.expt_name,
            action_space=action_space,
            observation_space=observation_space,
            connstr=self.db_config.get_uri(),
            workloads=self.workload,
            seed=self.seed,
            setup_sqls=self.setup_sql,
            timeout_s=5 * 60,
        )

        observation, info = env.reset(seed=15721)
        df = observation_space.convert_observations_to_df(observation)
        df.to_parquet(self.pq_path)
        # TODO(WAN): tuning...
        env.close()

        if self.should_nyoom:
            nyoom_stop(self.db_config)
