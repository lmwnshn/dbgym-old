# TODO(WAN):
#  Bring back all the scheduling and PREPARE stuff that made sense in an OLTP world.
#  Bring back the [historical, future] workload split if we're trying to do forecasting.

from pathlib import Path
from dbgym.env.dbgym import DbGymEnv

import copy
import pglast
import gymnasium
from dbgym.config import Config
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.space.action.fake_index import FakeIndexSpace
from dbgym.space.observation.qppnet.features import QPPNetFeatures
from dbgym.trainer.postgres import PostgresTrainer
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine.base import Connection
from sqlalchemy.engine.reflection import Inspector

from sklearn.model_selection import train_test_split

from dbgym.workload.workload import WorkloadTPCH

from autogluon.tabular import TabularDataset, TabularPredictor

import pandas as pd
import numpy as np


def _exec(conn: Connection, sql: str, verbose=True):
    if verbose:
        print(sql)
    conn.execute(text(sql))


def pgtune():
    # TODO(WAN): make this something you can pass in.
    pgconf_laptop = [
        # DB Version: 15
        # OS Type: linux
        # DB Type: dw
        # Total Memory (RAM): 16 GB
        # CPUs num: 8
        # Connections num: 20
        # Data Storage: ssd
        "ALTER SYSTEM SET max_connections = '20';",
        "ALTER SYSTEM SET shared_buffers = '4GB';",
        "ALTER SYSTEM SET effective_cache_size = '12GB';",
        "ALTER SYSTEM SET maintenance_work_mem = '2GB';",
        "ALTER SYSTEM SET checkpoint_completion_target = '0.9';",
        "ALTER SYSTEM SET wal_buffers = '16MB';",
        "ALTER SYSTEM SET default_statistics_target = '500';",
        "ALTER SYSTEM SET random_page_cost = '1.1';",
        "ALTER SYSTEM SET effective_io_concurrency = '200';",
        "ALTER SYSTEM SET work_mem = '26214kB';",
        "ALTER SYSTEM SET min_wal_size = '4GB';",
        "ALTER SYSTEM SET max_wal_size = '16GB';",
        "ALTER SYSTEM SET max_worker_processes = '8';",
        "ALTER SYSTEM SET max_parallel_workers_per_gather = '4';",
        "ALTER SYSTEM SET max_parallel_workers = '8';",
        "ALTER SYSTEM SET max_parallel_maintenance_workers = '4';",
    ]

    pgconf_dev8 = [
        # WARNING
        # this tool not being optimal
        # for very high memory systems

        # DB Version: 15
        # OS Type: linux
        # DB Type: dw
        # Total Memory (RAM): 188 GB
        # CPUs num: 80
        # Connections num: 20
        # Data Storage: ssd
        "ALTER SYSTEM SET max_connections = '20';",
        "ALTER SYSTEM SET shared_buffers = '47GB';",
        "ALTER SYSTEM SET effective_cache_size = '141GB';",
        "ALTER SYSTEM SET maintenance_work_mem = '2GB';",
        "ALTER SYSTEM SET checkpoint_completion_target = '0.9';",
        "ALTER SYSTEM SET wal_buffers = '16MB';",
        "ALTER SYSTEM SET default_statistics_target = '500';",
        "ALTER SYSTEM SET random_page_cost = '1.1';",
        "ALTER SYSTEM SET effective_io_concurrency = '200';",
        "ALTER SYSTEM SET work_mem = '30801kB';",
        "ALTER SYSTEM SET min_wal_size = '4GB';",
        "ALTER SYSTEM SET max_wal_size = '16GB';",
        "ALTER SYSTEM SET max_worker_processes = '80';",
        "ALTER SYSTEM SET max_parallel_workers_per_gather = '40';",
        "ALTER SYSTEM SET max_parallel_workers = '80';",
        "ALTER SYSTEM SET max_parallel_maintenance_workers = '4';",
    ]
    engine = create_engine(Config.TRAINER_PG_URI, execution_options={"isolation_level": "AUTOCOMMIT"})
    with engine.connect() as conn:
        for sql in pgconf_laptop:
            _exec(conn, sql)


def load(dataset):
    # TODO(WAN): support for more datasets.
    if dataset == "tpch":
        load_tpch()
    else:
        raise RuntimeError(f"{dataset=} not supported.")


def load_tpch():
    tpch_sf = Path("/tpch_sf1").absolute()
    engine = create_engine(Config.TRAINER_PG_URI, execution_options={"isolation_level": "AUTOCOMMIT"})
    with engine.connect() as conn:
        tables = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
        with open(Path("/tpch_schema") / "tpch_schema.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
        for table in tables:
            _exec(conn, f"TRUNCATE {table} CASCADE")
        for table in tables:
            table_path = tpch_sf / f"{table}.tbl"
            _exec(conn, f"COPY {table} FROM '{str(table_path)}' CSV DELIMITER '|'")
        with open(Path("/tpch_schema") / "tpch_constraints.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)


def prewarm_all():
    engine = create_engine(Config.TRAINER_PG_URI, execution_options={"isolation_level": "AUTOCOMMIT"})
    inspector: Inspector = inspect(engine)
    with engine.connect() as conn:
        _exec(conn, "CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        for table in inspector.get_table_names():
            _exec(conn, f"SELECT pg_prewarm('{table}')")
            for index in inspector.get_indexes(table):
                index_name = index["name"]
                _exec(conn, f"SELECT pg_prewarm('{index_name}')")


def vacuum_analyze_all():
    engine = create_engine(Config.TRAINER_PG_URI, execution_options={"isolation_level": "AUTOCOMMIT"})
    inspector: Inspector = inspect(engine)
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            _exec(conn, f"VACUUM ANALYZE {table}")


def prepare():
    # TODO(WAN):
    #  I'm back in an OLAP world, so I only need to run this once at the start of the gym.
    #  Otherwise, I need to bring back:
    #  - PREPARE for high throughput OLTP.
    #  - Reloading state between gym iterations, arguably you want to push this logic down further.
    #  Code that does this exists as of time of writing (see meow or nyoom branch), I just hate it.
    vacuum_analyze_all()
    prewarm_all()


def gym(name, workloads, seed=15721):
    # Run the queries in the gym.
    engine = create_engine(Config.TRAINER_PG_URI, execution_options={"isolation_level": "AUTOCOMMIT"})
    # TODO(WAN): assumes read-only.
    db_snapshot_path = Path("/dbgym/snapshot.pkl").absolute()
    if not db_snapshot_path.exists():
        db_snapshot = DatabaseSnapshot(engine)
        db_snapshot.to_file(db_snapshot_path)
    db_snapshot = DatabaseSnapshot.from_file(db_snapshot_path)

    action_space = FakeIndexSpace(1)
    observation_space = QPPNetFeatures(db_snapshot=db_snapshot, seed=seed)

    # noinspection PyTypeChecker
    env: DbGymEnv = gymnasium.make(
        "dbgym/DbGym-v0",
        # disable_env_checker=True,
        action_space=action_space,
        observation_space=observation_space,
        connstr=Config.TRAINER_PG_URI,
        workloads=workloads,
        seed=seed,
    )

    obs_path = Path(f"/dbgym/{name}/obs/").absolute()
    obs_path.mkdir(parents=True, exist_ok=True)
    obs_iter = 0

    observation, info = env.reset(seed=15721)
    df = observation_space.convert_observations_to_df(observation)
    df.to_parquet(obs_path / f"{obs_iter}.parquet")
    obs_iter += 1

    # TODO(WAN): eventually...
    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     train_df = observation_space.convert_observations_to_df(observation)
    #     train_df.to_parquet(obs_path / f"{obs_iter}.parquet")
    #     obs_iter += 1
    #
    #     if terminated or truncated:
    #         observation, info = env.reset()
    env.close()


def hack_tablesample_tpch(workload: WorkloadTPCH):
    result = copy.deepcopy(workload)
    hack_nation = ["nation n1", "nation n2"]
    hack_lineitem = ["lineitem l1", "lineitem l2", "lineitem l3"]
    tables = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]

    for i, query in enumerate(result.queries):
        query = query.replace("select\n\tnation", "select\n\tNATION")
        query = query.replace("group by\n\tnation", "group by\n\tNATION")
        query = query.replace("order by\n\tnation", "order by\n\tNATION")
        query = query.replace("n_name as nation", "n_name as NATION")
        query = query.replace("as c_orders", "as C_ORDERS")
        # This appears to be a bug in the PostgreSQL optimizer, TPC-H Q20 samplescan suffers.
        query = query.replace("from\n\t\t\t\t\tlineitem", "from\n\t\t\t\t\tLINEITEM")
        query = query.replace("lineitem l3", "LINEITEM l3")

        old_query = query
        for table in hack_nation:
            query = query.replace(f"{table} ", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721) ")
            query = query.replace(f"{table},", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721),")
            query = query.replace(f"{table}\n", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721)\n")
        skip_nation = old_query != query

        old_query = query
        for table in hack_lineitem:
            query = query.replace(f"{table} ", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721) ")
            query = query.replace(f"{table},", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721),")
            query = query.replace(f"{table}\n", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721)\n")
        skip_lineitem = old_query != query

        for table in tables:
            if table == "nation" and skip_nation:
                continue
            if table == "lineitem" and skip_lineitem:
                continue
            query = query.replace(f"{table} ", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721) ")
            query = query.replace(f"{table},", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721),")
            query = query.replace(f"{table}\n", f"{table} TABLESAMPLE BERNOULLI (10) REPEATABLE (15721)\n")

        query = query.replace("LINEITEM", "lineitem")
        query = query.replace("NATION", "nation")
        query = query.replace("as C_ORDERS", "as c_orders")
        result.queries[i] = query
    return result


def main():
    workload_seed_start, workload_seed_end = 15721, 15725
    workloads = [WorkloadTPCH(seed) for seed in range(workload_seed_start, workload_seed_end + 1)]
    train_workloads, test_workloads = train_test_split(workloads, test_size=0.2)
    tablesample_workloads = [hack_tablesample_tpch(workload) for workload in train_workloads]

    seed = 15721
    with PostgresTrainer(Config.TRAINER_URL) as trainer:
        assert trainer.dbms_exists(), "Startup failed?"
        pgtune()
        trainer.dbms_restart()
        load("tpch")
        prepare()
        gym("test", test_workloads, seed=seed)
        gym("train", train_workloads, seed=seed)
        gym("tablesample", tablesample_workloads, seed=seed)

    def flatten(df):
        flatteneds = []
        for col in ["Children Observation Indexes", "Features", "Query Hash"]:
            col_df = pd.DataFrame(df[col].tolist(), index=df.index)
            col_df = col_df.rename(columns=lambda num: f"{col}_{num}")
            flatteneds.append(col_df)
        for col in ["Node Type", "Observation Index", "Query Num", "Actual Total Time (us)"]:
            flatteneds.append(df[col])
        return pd.concat(flatteneds, axis=1)

    test_df = pd.read_parquet("/dbgym/test/obs/0.parquet")
    train_df = pd.read_parquet("/dbgym/train/obs/0.parquet")
    tablesample_df = pd.read_parquet("/dbgym/tablesample/obs/0.parquet")
    test_data = TabularDataset(flatten(test_df))
    train_data = TabularDataset(flatten(train_df))
    tablesample_data = TabularDataset(flatten(tablesample_df))
    save_path = "/dbgym/models/autogluon/"
    # noinspection PyUnreachableCode
    if True:
        predictor = TabularPredictor(label="Actual Total Time (us)", path=save_path).fit(train_data, time_limit=30)
    else:
        predictor = TabularPredictor.load(save_path)
    y_pred = predictor.predict(test_data)
    eval_df = pd.concat([y_pred, test_data["Actual Total Time (us)"]], axis=1)
    eval_df.columns = ["Predicted Latency (us)", "Actual Latency (us)"]
    metrics_df = eval_df.copy()
    metrics_df["diff (us)"] = (metrics_df["Predicted Latency (us)"] - metrics_df["Actual Latency (us)"]).abs()
    metrics_df["q_err"] = np.nan_to_num(
        np.maximum(
            metrics_df["Predicted Latency (us)"] / metrics_df["Actual Latency (us)"],
            metrics_df["Actual Latency (us)"] / metrics_df["Predicted Latency (us)"]
        ),
        nan=np.inf
    )
    print(metrics_df["diff (us)"].describe())
    print(metrics_df["q_err"].describe())
    print()


if __name__ == "__main__":
    main()

