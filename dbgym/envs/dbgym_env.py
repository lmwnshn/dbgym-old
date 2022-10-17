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
from dbgym.spaces.observations.qppnet.features import QPPNetFeatures


class DbGymEnv(gym.Env):
    def __init__(self, gym_spec: GymSpec, seed=15721, try_prepare=False, print_errors=False):
        self._rng = np.random.default_rng(seed=seed)
        self._gym_spec = gym_spec
        self._trainer = PostgresTrainer(gym_spec=self._gym_spec, seed=seed, hack=True)

        self._trainer.create_target_dbms()
        # Track whether the DB is dirty, i.e., needs to be recreated to get back to original state.
        self._db_dirty = False

        assert len(self._gym_spec.snapshot) == 1, "We only support one schema right now."
        self.action_space = IndexSpace(gym_spec=gym_spec, seed=seed)
        self.observation_space = QPPNetFeatures(gym_spec=gym_spec, seed=seed)

        self._runner = WorkloadRunner()
        self._try_prepare = try_prepare
        self._print_errors = print_errors

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
        if self._db_dirty:
            self._trainer.create_target_dbms()
            self._db_dirty = False
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
        for i, workload in enumerate(self._gym_spec.historical_workloads):
            workload_db_path = workload._workload_path
            print(f"Collecting observations for [{i}/{len(self._gym_spec.historical_workloads)}]: {workload_db_path}")
            observation, info = self._runner.run(
                workload_db_path,
                engine,
                self.observation_space,
                current_observation_idx,
                print_errors=self._print_errors,
                try_prepare=self._try_prepare,
            )
            current_observation_idx += len(observation)
            assert type(observation) == list, "TODO hacks"
            observations.extend(observation)
            if info != {}:
                infos[workload_db_path] = info
        self._db_dirty = True
        return observations, infos
