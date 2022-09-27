from typing import Optional, Tuple

import gym
import numpy as np
from gym.core import ActType, ObsType

from dbgym.spaces.index import IndexSpace
from dbgym.spaces.qppnet import QPPNetFeatures

from dbgym.envs.workload_runner import WorkloadRunner
from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.trainer import PostgresTrainer

from sqlalchemy.pool import NullPool
from sqlalchemy.engine import create_engine

import time

class DbGymEnv(gym.Env):
    def __init__(self, gym_spec: GymSpec, seed=15721):
        self._rng = np.random.default_rng(seed=seed)
        self._gym_spec = gym_spec
        assert len(gym_spec.snapshot) == 1, "We only support one schema right now."
        self.action_space = IndexSpace(gym_spec=gym_spec, seed=seed)
        self.observation_space = QPPNetFeatures(gym_spec=gym_spec, seed=seed)
        self._trainer = PostgresTrainer(gym_spec=self._gym_spec, seed=seed)
        self._runner = WorkloadRunner()

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
        self._trainer.delete_target_dbms()
        self._trainer.create_target_dbms()
        observation, info = self._run_workload()
        print("_reset finished at ", time.time())
        return observation, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Play through the entire workload.
        observation, info = self._run_workload()
        reward = 0.0
        terminated, truncated = True, False
        return observation, reward, terminated, truncated, info

    def _run_workload(self) -> Tuple[ObsType, dict]:
        engine = create_engine(
            self._trainer.get_target_dbms_connstr_sqlalchemy(), poolclass=NullPool,
        )
        workload_db_path = self._gym_spec.historical_workload._workload_path
        observation, info = self._runner.run(workload_db_path, engine, self.observation_space)
        print("_run_workload finished at ", time.time())
        return observation, info
