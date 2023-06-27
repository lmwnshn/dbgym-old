import time
from threading import Lock, Thread
from typing import Optional, Tuple

import gymnasium
import numpy as np
import psycopg
from dbgym.space.observation.base import BaseFeatureSpace
from dbgym.space.observation.qppnet.features import QPPNetFeatures
from dbgym.workload.workload import Workload
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import Space
from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.exc import OperationalError
from tqdm import tqdm


class DbGymEnv(gymnasium.Env):
    def __init__(
        self,
        name: str,
        action_space: Space,
        observation_space: Space,
        workloads: list[Workload],
        connstr: str,
        seed=15721,
        timeout_s=1,
        setup_sqls: list[str] = None,
    ):
        self._rng = np.random.default_rng(seed=seed)

        self.name = name
        self.action_space = action_space
        self.observation_space = observation_space
        self.workloads = workloads
        self.connstr = connstr
        self.seed = seed
        self.timeout_s = timeout_s
        self.setup_sqls = setup_sqls
        self._engine = create_engine(
            self.connstr, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[ObsType, dict]:
        # Reset the RNG.
        if seed is None:
            seed = self.seed
        self._rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        observation, info = self._run_workload()
        return observation, info

    # noinspection PyUnreachableCode
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        # Play through the entire workload.
        raise RuntimeError(f"{action=} not supported yet.")
        observation, info = self._run_workload()
        reward = 0.0
        terminated, truncated = True, False
        return observation, reward, terminated, truncated, info

    def _run_workload(self) -> Tuple[ObsType, dict]:
        assert isinstance(self.observation_space, BaseFeatureSpace)

        runner_pid = 0
        cancel_lock = Lock()
        cancel_thread_run = True
        cancel_in_query = False
        cancel_start_time = 0

        def cancel_func():
            while cancel_thread_run:
                time.sleep(min(5, self.timeout_s))
                with cancel_lock:
                    if cancel_in_query:
                        elapsed_time_s = time.monotonic() - cancel_start_time
                        if elapsed_time_s > self.timeout_s:
                            with self._engine.connect() as cancel_conn:
                                cancel_conn.execute(text(f"SELECT pg_cancel_backend({runner_pid})"))

        cancel_thread = Thread(target=cancel_func)
        cancel_thread.start()

        observations = []
        infos = {}
        query_num = 0
        obs_idx = 0

        with self._engine.connect() as conn:
            for sql in self.setup_sqls:
                results = conn.execute(text(sql))
            runner_pid = int(conn.execute(text("SELECT pg_backend_pid()")).fetchone()[0])
            for workload in tqdm(self.workloads, desc=f"{self.name}: Iterating over workloads.", leave=None):
                for sql_text in tqdm(workload.queries, desc=f"{self.name}: Running queries in workload.", leave=None):
                    try:
                        query_num += 1
                        # TODO(WAN): hack!!
                        can_prefix = sql_text.strip().lower().split()[0] in ["delete", "insert", "select", "update"]
                        if can_prefix:
                            sql = text(self.observation_space.SQL_PREFIX + sql_text)
                        else:
                            sql = text(sql_text)

                        with cancel_lock:
                            cancel_in_query = True
                            cancel_start_time = time.monotonic()
                        results = conn.execute(sql)
                        with cancel_lock:
                            cancel_in_query = False
                            cancel_start_time = 0
                        assert results.returns_rows if can_prefix else True, f"What happened? SQL: {sql}"
                        if results.returns_rows:
                            results = results.fetchall()
                            if can_prefix and isinstance(self.observation_space, QPPNetFeatures):
                                assert len(results) == 1, "Multi-query SQL?"
                                assert len(results[0]) == 1, "Multi-column result for EXPLAIN?"
                                result_dicts = results[0][0]
                                for result_dict in result_dicts:
                                    new_observations = self.observation_space.generate(
                                        result_dict,
                                        query_num,
                                        obs_idx,
                                        str(sql),
                                    )
                                    obs_idx += len(new_observations)
                                    observations.extend(new_observations)
                    except OperationalError as e:
                        try:
                            raise e.orig
                        except psycopg.errors.QueryCanceled:
                            with cancel_lock:
                                cancel_in_query = False
                                cancel_start_time = 0

        with cancel_lock:
            cancel_thread_run = False
        cancel_thread.join()
        return observations, infos
