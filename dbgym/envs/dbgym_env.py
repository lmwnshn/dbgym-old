import time
from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType
from sqlalchemy.engine import create_engine
from sqlalchemy.pool import NullPool

from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.trainer import PostgresTrainer
from dbgym.envs.workload_runner import WorkloadRunner
from dbgym.spaces.index import IndexSpace
from dbgym.spaces.qppnet_features import QPPNetFeatures


class DbGymEnv(gym.Env):
    def __init__(self, gym_spec: GymSpec, seed=15721):
        self._rng = np.random.default_rng(seed=seed)
        self._gym_spec = gym_spec
        self._trainer = PostgresTrainer(gym_spec=self._gym_spec, seed=seed)
        assert (
            len(self._gym_spec.snapshot) == 1
        ), "We only support one schema right now."
        self._runner = WorkloadRunner()
        self.action_space = IndexSpace(gym_spec=gym_spec, seed=seed)
        self.observation_space = QPPNetFeatures(gym_spec=gym_spec, seed=seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        # Reset the RNG.
        self._rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self._trainer.create_target_dbms()
        observation, info = self._run_workload()
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Play through the entire workload.
        observation, info = self._run_workload()
        reward = 0.0
        terminated, truncated = True, False
        return observation, reward, terminated, truncated, info

    def _run_workload(self) -> Tuple[ObsType, dict]:
        engine = create_engine(
            self._trainer.get_target_dbms_connstr_sqlalchemy(),
            poolclass=NullPool,
        )
        observations = []
        infos = {}
        current_observation_idx = 0
        for workload in self._gym_spec.historical_workloads:
            workload_db_path = workload._workload_path
            print(f"Collecting observations for: {workload_db_path}")
            observation, info = self._runner.run(
                workload_db_path,
                engine,
                self.observation_space,
                current_observation_idx,
            )
            current_observation_idx += len(observation)
            assert type(observation) == list, "TODO hacks"
            observations.extend(observation)
            if info != {}:
                infos[workload_db_path] = info
        return observations, infos
