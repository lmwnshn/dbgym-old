# TODO(WAN):
#  Bring back all the scheduling and PREPARE stuff that made sense in an OLTP world.
#  Bring back the [historical, future] workload split if we're trying to do forecasting.
from pathlib import Path

import pglast
from dbgym.config import Config
from dbgym.gym_config import GymConfig
from dbgym.gym_config_model import GymConfigModel
from dbgym.gym_config_plot import GymConfigPlot
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.trainer.postgres import PostgresTrainer
from dbgym.workload.workload import (
    Workload,
    WorkloadNaiveTablesampleTPCH,
    WorkloadSmartTablesampleTPCH,
    WorkloadTPCH,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import NullPool, create_engine, inspect, text
from sqlalchemy.engine.base import Connection
from sqlalchemy.engine.reflection import Inspector


def _exec(conn: Connection, sql: str, verbose=True):
    if verbose:
        print(sql)
    conn.execute(text(sql))


def pgtune():
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        for sql in Config.PGTUNE_STATEMENTS:
            _exec(conn, sql)
    engine.dispose()


def exists(dataset):
    # TODO(WAN): this is just a convenient hack based on testing the last thing that load does.
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        if dataset == "tpch":
            res = conn.execute(text("SELECT * FROM pg_indexes WHERE indexname = 'l_sk_pk'")).fetchall()
            return len(res) > 0
        else:
            raise RuntimeError(f"{dataset=} not supported.")
    engine.dispose()


def load_if_not_exists(dataset):
    if exists(dataset):
        return False
    else:
        load(dataset)
        return True


def load(dataset):
    # TODO(WAN): support for more datasets.
    if dataset == "tpch":
        load_tpch()
    elif dataset == "dsb":
        load_dsb()
    else:
        raise RuntimeError(f"{dataset=} not supported.")


def load_tpch():
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        tables = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
        with open(Path("/tpch_schema") / "tpch_schema.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
        for table in tables:
            _exec(conn, f"TRUNCATE {table} CASCADE")
        for table in tables:
            table_path = Config.TPCH_DATA / f"{table}.tbl"
            _exec(conn, f"COPY {table} FROM '{str(table_path)}' CSV DELIMITER '|'")
        with open(Path("/tpch_schema") / "tpch_constraints.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
    engine.dispose()


def load_dsb():
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        tables = ["region", "nation", "part", "supplier", "partsupp", "customer", "orders", "lineitem"]
        with open(Path("/tpch_schema") / "tpch_schema.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
        for table in tables:
            _exec(conn, f"TRUNCATE {table} CASCADE")
        for table in tables:
            table_path = Config.TPCH_DATA / f"{table}.tbl"
            _exec(conn, f"COPY {table} FROM '{str(table_path)}' CSV DELIMITER '|'")
        with open(Path("/tpch_schema") / "tpch_constraints.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
    engine.dispose()


def prewarm_all():
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    inspector: Inspector = inspect(engine)
    with engine.connect() as conn:
        _exec(conn, "CREATE EXTENSION IF NOT EXISTS pg_prewarm")
        for table in inspector.get_table_names():
            _exec(conn, f"SELECT pg_prewarm('{table}')")
            for index in inspector.get_indexes(table):
                index_name = index["name"]
                _exec(conn, f"SELECT pg_prewarm('{index_name}')")
    engine.dispose()


def vacuum_analyze_all():
    engine = create_engine(
        Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    inspector: Inspector = inspect(engine)
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            # _exec(conn, f"VACUUM FULL ANALYZE {table}")
            _exec(conn, f"VACUUM ANALYZE {table}")
    engine.dispose()


def prepare():
    # TODO(WAN):
    #  I'm back in an OLAP world, so I only need to run this once at the start of the gym.
    #  Otherwise, I need to bring back:
    #  - PREPARE for high throughput OLTP.
    #  - Reloading state between gym iterations, arguably you want to push this logic down further.
    #  Code that does this exists as of time of writing (see meow or nyoom branch), I just hate it.
    vacuum_analyze_all()
    prewarm_all()


GYM_CONFIGS = []


def setup_gym_configs_tpch():
    global GYM_CONFIGS

    # Setup TPC-H worklaads.
    workload_seed_start, workload_seed_end = Config.WORKLOAD_SEED_START, Config.WORKLOAD_SEED_END
    workloads = [WorkloadTPCH(seed) for seed in range(workload_seed_start, workload_seed_end + 1)]
    default_workload, test_workload = train_test_split(workloads, test_size=0.2)
    seed = 15721
    should_overwrite = False

    gym_configs = [
        GymConfig(
            name="tpch_test",
            workload=test_workload,
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            name="tpch_default",
            workload=default_workload,
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            name="tpch_naive_sql",
            workload=[WorkloadNaiveTablesampleTPCH(workload) for workload in default_workload],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            name="tpch_smart_sql",
            workload=[WorkloadSmartTablesampleTPCH(workload) for workload in default_workload],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            name="tpch_tskip_sample",
            workload=default_workload,
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[
                "CREATE EXTENSION IF NOT EXISTS nyoom",
                f"SET nyoom.should_sample_seq = true",
                f"SET nyoom.sample_seq_pct = 10",
                f"SET nyoom.sample_seq_seed = 15721",
            ],
            nyoom_args=None,
        ),
        GymConfig(
            name="tpch_tskip_cutoff",
            workload=default_workload,
            seed=seed,
            should_nyoom=True,
            should_overwrite=should_overwrite,
            setup_sql=[
                "CREATE EXTENSION IF NOT EXISTS nyoom",
                f"SET nyoom.should_sample_seq = false",
                f"SET nyoom.telemetry_window_size = 10000",
                f"SET nyoom.telemetry_tuple_count = 25000",
            ],
            nyoom_args={"method": "tskip", "tskip_wiggle_std": 2.0, "tskip_wiggle_sampen": 20},
        ),
        GymConfig(
            name="tpch_tskip_sample_cutoff",
            workload=default_workload,
            seed=seed,
            should_nyoom=True,
            should_overwrite=should_overwrite,
            setup_sql=[
                "CREATE EXTENSION IF NOT EXISTS nyoom",
                f"SET nyoom.should_sample_seq = true",
                f"SET nyoom.sample_seq_pct = 10",
                f"SET nyoom.sample_seq_seed = 15721",
                f"SET nyoom.telemetry_window_size = 10000",
                f"SET nyoom.telemetry_tuple_count = 25000",
            ],
            nyoom_args={"method": "tskip", "tskip_wiggle_std": 2.0, "tskip_wiggle_sampen": 20},
        ),
    ]

    sequential_hack = True
    for gym_config in gym_configs:
        if sequential_hack:
            gym_config.setup_sql.append("set max_parallel_workers_per_gather = 0")
        GYM_CONFIGS.append(gym_config)


def run_gym_config(gym_config: GymConfig):
    db_snapshot_path = Path("/dbgym/snapshot.pkl").absolute()

    if not gym_config.should_run():
        print(f"Skipping: {gym_config.name}")
        return

    print(f"Running: {gym_config.name}")
    # TODO(WAN): previously we ran the gym multiple times for each initialization of the database, but vacuum etc.
    with PostgresTrainer(Config.TRAINER_URL, force_rebuild=False) as trainer:
        assert trainer.dbms_exists(), "Startup failed?"
        trainer.dbms_install_nyoom()
        pgtune()
        trainer.dbms_restart()
        newly_loaded = load_if_not_exists("tpch")
        prepare()

        # TODO(WAN): assumes read-only.
        if newly_loaded or not db_snapshot_path.exists():
            print("Snapshot: generating.")
            engine = create_engine(
                Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            db_snapshot = DatabaseSnapshot(engine)
            engine.dispose()
            db_snapshot.to_file(db_snapshot_path)
            print("Snapshot: complete.")
        db_snapshot = DatabaseSnapshot.from_file(db_snapshot_path)

        gym_config.force_run(Config.TRAINER_PG_URI, db_snapshot)


def main():
    global GYM_CONFIGS

    setup_gym_configs_tpch()
    for gym_config in GYM_CONFIGS:
        run_gym_config(gym_config)

    test_df_name = "tpch_test"
    train_df_names = [
        "tpch_default",
        "tpch_naive_sql",
        "tpch_smart_sql",
        "tpch_tskip_sample",
        "tpch_tskip_cutoff",
        "tpch_tskip_sample_cutoff",
    ]

    GymConfigModel.generate_model(test_df_name=test_df_name, train_df_names=train_df_names)
    GymConfigPlot.generate_plot(
        plot_suffix="tpch",
        expt_names=train_df_names,
        plot_names=train_df_names,
    )
    GymConfigPlot.generate_plot_runtime_by_operator(
        plot_suffix="tpch",
        expt_names=train_df_names,
        plot_names=train_df_names,
    )


if __name__ == "__main__":
    main()
