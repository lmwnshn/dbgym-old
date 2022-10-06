import threading
import time
from abc import ABC
from pathlib import Path
from queue import Queue
from typing import Optional

import numpy as np
import pandas as pd
import psutil
from gym.core import ObsType
from gym.spaces import Space
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, CursorResult, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import SingletonThreadPool
from tqdm import tqdm

from dbgym.spaces.qppnet_features import QPPNetFeatures
from dbgym.util.sql import substitute


class Work(ABC):
    def __init__(self, work_query_num: int, sql_keyword: str):
        # work_query_num tracks the ORIGINAL workload.db query number.
        self.work_query_num = work_query_num
        self.sql_keyword = sql_keyword.lower()

    def execute(
        self, conn: Connection, sql_prefix: Optional[str] = None
    ) -> (CursorResult, bool):
        raise NotImplementedError

    def try_add_prefix(self, sql_prefix, sql):
        if (
            self.sql_keyword not in ["delete", "insert", "select", "update"]
            or sql_prefix is None
        ):
            return sql, False
        return f"{sql_prefix} {sql}", True


class WorkQueryString(Work):
    def __init__(self, work_query_num: int, query: str, sql_keyword: str):
        super().__init__(work_query_num, sql_keyword)
        self.query = query

    def __str__(self):
        return f"WQS[{self.query}]"

    def execute(
        self, conn: Connection, sql_prefix: Optional[str] = None
    ) -> (CursorResult, bool):
        sql, prefixed = self.try_add_prefix(sql_prefix, self.query)
        return conn.execute(sql), prefixed


class WorkQueryPrepare(Work):
    def __init__(
        self,
        work_query_num: int,
        template_id: int,
        params: Optional[tuple],
        sql_keyword: str,
    ):
        super().__init__(work_query_num, sql_keyword)
        self.template_id = template_id
        self.params = params

    def __str__(self):
        return f"WQP[{self.template_id}, {self.params}]"

    def execute(
        self, conn: Connection, sql_prefix: Optional[str] = None
    ) -> (CursorResult, bool):
        sql = f"EXECUTE work_{self.template_id}{self._format_params(self.params)}"
        sql, prefixed = self.try_add_prefix(sql_prefix, sql)
        return conn.execute(sql), prefixed

    def _format_params(self, params: Optional[tuple]) -> str:
        if params is None:
            return ""
        return "(" + ",".join(params) + ")"


def _should_prepare(template):
    startswith = [
        "alter ",
        "set ",
        "show ",
    ]
    contains = [
        "pg_catalog",
    ]
    matches = [
        "begin",
        "commit",
        "rollback",
    ]
    template = template.strip().lower()
    do_not_prepare = template in matches
    do_not_prepare = do_not_prepare or any(
        template.startswith(prefix) for prefix in startswith
    )
    do_not_prepare = do_not_prepare or any(substr in template for substr in contains)
    return not do_not_prepare


def _generate_work(
    work_query_num: int,
    prepare_types: dict[int, list[str]],
    all_templates: dict[int, str],
    incompatible_templates: dict[int, str],
    template_id: int,
    params: tuple = None,
):
    if template_id in prepare_types and params is not None:
        new_params = []
        types = prepare_types[template_id]
        assert len(types) == len(params), f"Invalid parameters? {types}, {params}"
        for param_type, param in zip(types, params):
            if param_type in [
                "smallint",
                "integer",
                "bigint",
                "decimal",
                "numeric",
                "real",
                "double precision",
                "smallserial",
                "serial",
                "bigserial",
            ]:
                param = param[1:-1]
            new_params.append(param)
        params = tuple(new_params)

    template = all_templates[template_id].lower()
    if template.startswith("explain "):
        # Try extracting the query to override the EXPLAIN options.
        for keyword in ["delete", "insert", "select", "update"]:
            idx = template.find(keyword)
            if idx != -1:
                template = template[idx:]
                break
        else:
            raise RuntimeError(f"Add parsing logic for this: {template}")
    sql_keyword = template.split(" ")[0]

    if template_id not in incompatible_templates:
        # This template was PREPARE'd.
        return WorkQueryPrepare(work_query_num, template_id, params, sql_keyword)
    # This template could not be PREPARE'd.
    template = incompatible_templates[template_id]
    if params is None:
        # There are no parameters.
        return WorkQueryString(work_query_num, template, sql_keyword)
    # There are parameters; make an expensive call to substitute().
    params = {f"${i}": param for i, param in enumerate(params, 1)}
    return WorkQueryString(work_query_num, substitute(template, params), sql_keyword)


def _submission_worker(
    workload_db_path,
    prepare_queue,
    prepare_types_queue,
    work_queue,
    prepare_event,
    prepare_types_event,
    done_event,
):
    # Submit work to the queues.

    # Read directly from the DB file.
    engine = create_engine(
        f"sqlite:///{workload_db_path}",
        poolclass=SingletonThreadPool,
        pool_size=psutil.cpu_count(),
    )
    # TODO(WAN): Figure out the semantics of elapsed_s. Do we care? Or replay goes brr?
    min_query_num = engine.execute("select min(query_num) from workload").fetchone()[0]
    max_query_num = engine.execute("select max(query_num) from workload").fetchone()[0]
    total_query_num = engine.execute("select count(*) from workload").fetchone()[0]

    # PREPARE any templates that can be prepared.
    all_templates: dict[int, str] = {}
    incompatible_templates: dict[int, str] = {}
    sql = "select id, template from query_template"
    results = engine.execute(sql).fetchall()
    for result in results:
        template_id, template = result
        all_templates[template_id] = template
        if _should_prepare(template):
            prepare_queue.put((template_id, template))
        else:
            assert template_id not in incompatible_templates, "Duplicate template ID?"
            incompatible_templates[template_id] = template
    prepare_event.set()

    # Wait for all the PREPARE types to come in.
    while not prepare_types_event.wait(1):
        pass
    prepare_types: dict[int, list[str]] = {}
    while not prepare_types_queue.empty():
        template_id, types = prepare_types_queue.get()
        prepare_types[template_id] = types
        prepare_types_queue.task_done()

    # Get templates without parameters.
    sql = (
        "select distinct(template_id) "
        "from workload where params_id is null "
        "order by template_id"
    )
    templates_without_params = engine.execute(sql).fetchall()
    # Get templates with parameters.
    sql = (
        "select distinct(template_id) "
        "from workload where params_id is not null "
        "order by template_id"
    )
    templates_with_params = engine.execute(sql).fetchall()

    # HACK: Use prepare queue to communicate total.
    prepare_queue.put(total_query_num)

    cur_query_num = min_query_num
    tick_query_num = 50000
    while True:
        all_results = []
        # Process templates without parameters.
        for template_id, *_ in templates_without_params:
            sql = (
                "select w.query_num, w.template_id "
                f"from workload w where "
                f"{cur_query_num} <= query_num and "
                f"query_num < {cur_query_num + tick_query_num} and "
                f"w.template_id = {template_id} "
                "order by w.query_num"
            )
            results = engine.execute(sql).fetchall()
            results = [
                (
                    qnum,
                    _generate_work(
                        qnum,
                        prepare_types,
                        all_templates,
                        incompatible_templates,
                        template_id,
                        params=None,
                    ),
                )
                for qnum, template_id in results
            ]
            all_results.extend(results)
        # Process templates with parameters.
        for template_id, *_ in templates_with_params:
            sql = (
                "select w.query_num, w.template_id, p.* "
                f"from workload w, template_{template_id}_params p where "
                f"{cur_query_num} <= query_num and "
                f"query_num < {cur_query_num + tick_query_num} and "
                "w.params_id = p.id and "
                f"w.template_id = {template_id} "
                "order by w.query_num"
            )
            results = engine.execute(sql).fetchall()
            results = [
                (
                    qnum,
                    _generate_work(
                        qnum,
                        prepare_types,
                        all_templates,
                        incompatible_templates,
                        template_id,
                        params=params,
                    ),
                )
                for qnum, template_id, params_id, *params in results
            ]
            all_results.extend(results)
        # Sort by query number.
        all_results.sort(key=lambda x: x[0])
        # The initial smaller value is to give the other thread some work to do.

        for query_num, work in all_results:
            work_queue.put(work)

        cur_query_num = cur_query_num + tick_query_num
        tick_query_num = max(tick_query_num, 500000)
        if cur_query_num >= max_query_num:
            break
    done_event.set()


class WorkloadRunner:
    def __init__(self):
        pass

    def run(
        self,
        workload_db_path: Path,
        engine: Engine,
        obs_space: Space,
        current_observation_idx=0,
        print_errors=False,
    ) -> tuple[ObsType, dict]:
        sql_prefix = None
        if isinstance(obs_space, QPPNetFeatures):
            sql_prefix = "EXPLAIN (ANALYZE, FORMAT JSON, VERBOSE) "

        with engine.connect(close_with_result=False).execution_options(
            autocommit=False
        ) as conn:
            observations, info = [], {}
            # Start a worker thread.
            prepare_queue = Queue()
            work_queue: Queue[Work] = Queue()
            prepare_event = threading.Event()
            prepare_types_queue = Queue()
            prepare_types_event = threading.Event()
            done_event = threading.Event()
            submission_thread = threading.Thread(
                target=_submission_worker,
                args=(
                    workload_db_path,
                    prepare_queue,
                    prepare_types_queue,
                    work_queue,
                    prepare_event,
                    prepare_types_event,
                    done_event,
                ),
            )
            submission_thread.start()

            start_time = time.time()
            # TODO(WAN): exception handling for both threads.
            # Wait until all the PREPARE statements are generated.
            while not prepare_event.wait(1):
                pass
            # Execute the PREPARE statements.
            while not prepare_queue.empty():
                template_id, template = prepare_queue.get()
                sql = f"PREPARE work_{template_id} AS {template}"
                # print("PREPARE: ", sql)
                conn.execute(sql)
                prepare_queue.task_done()

            sql = "select name, parameter_types from pg_prepared_statements"
            results = conn.execute(sql)
            for result in results:
                name, types = result
                prefix, template_id = name.split("_")
                assert prefix == "work", f"What prepared statements got added? {result}"
                assert types.startswith("{") and types.endswith(
                    "}"
                ), f"Did they change the format? {types}"
                template_id = int(template_id)
                types = types[1:-1].split(",")
                prepare_types_queue.put((template_id, types))
            prepare_types_event.set()
            end_time = time.time()
            print(f"Prepare: {end_time - start_time:.3f}s")

            okay = 0
            errors = 0
            start_time = time.time()

            total_query_num = prepare_queue.get()
            with tqdm(total=total_query_num) as pbar:
                while True:
                    if done_event.is_set() and work_queue.empty():
                        break
                    work = work_queue.get()
                    try:
                        results, prefix_success = work.execute(conn, sql_prefix)
                        if results.returns_rows:
                            results = results.fetchall()
                            if isinstance(obs_space, QPPNetFeatures) and prefix_success:
                                # observations is a [query_plan] where query_plan = [plan_features_and_time],
                                # i.e., [[plan_features_and_time]]
                                assert len(results) == 1, "Multi-query SQL?"
                                assert (
                                    len(results[0]) == 1
                                ), "Multi-column result for EXPLAIN?"
                                result_dicts = results[0][0]
                                for result_dict in result_dicts:
                                    new_observations = obs_space.generate(
                                        result_dict,
                                        work.work_query_num,
                                        current_observation_idx,
                                    )
                                    current_observation_idx += len(new_observations)
                                    observations.extend(new_observations)
                        okay += 1
                    except SQLAlchemyError as e:
                        if print_errors:
                            print(f"WARNING: error executing {work}, error: {e}")
                        errors += 1
                    work_queue.task_done()
                    pbar.update()

            submission_thread.join()
            end_time = time.time()
            print(f"Execute [err {errors}, ok {okay}]: {end_time - start_time:.3f}s")
            return observations, info
