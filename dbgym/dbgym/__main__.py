# TODO(WAN):
#  Bring back all the scheduling and PREPARE stuff that made sense in an OLTP world.
#  Bring back the [historical, future] workload split if we're trying to do forecasting.
import os
from pathlib import Path

import pglast
from dbgym.config import Config
from dbgym.db_config import DbConfig, PgConfig
from dbgym.gym_config import GymConfig
from dbgym.gym_config_model import GymConfigModel
from dbgym.gym_config_plot import GymConfigPlot
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.trainer.postgres import PostgresTrainer
from dbgym.workload.workload import (
    Workload,
    WorkloadDSB,
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


def pgtune(gym_config: GymConfig):
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        for sql in Config.PGTUNE_STATEMENTS:
            _exec(conn, sql)
    engine.dispose()


def exists(gym_config: GymConfig):
    # TODO(WAN): this is just a convenient hack based on testing the last thing that load does.
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        dataset = gym_config.db_config.dataset
        if dataset == "tpch":
            res = conn.execute(text("SELECT * FROM pg_indexes WHERE indexname = 'l_sk_pk'")).fetchall()
            return len(res) > 0
        elif dataset == "dsb":
            res = conn.execute(
                text("SELECT * FROM pg_indexes WHERE indexname = '_dta_index_customer_5_949578421__k13_k5'")).fetchall()
            return len(res) > 0
        else:
            raise RuntimeError(f"{dataset=} not supported.")
    engine.dispose()


def load_if_not_exists(gym_config: GymConfig):
    if exists(gym_config):
        return False
    else:
        load(gym_config)
        return True


def load(gym_config: GymConfig):
    dataset = gym_config.db_config.dataset
    # TODO(WAN): support for more datasets.
    if dataset == "tpch":
        load_tpch(gym_config)
    elif dataset == "dsb":
        load_dsb(gym_config)
    else:
        raise RuntimeError(f"{dataset=} not supported.")


def load_tpch(gym_config: GymConfig):
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
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


def load_dsb(gym_config: GymConfig):
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with engine.connect() as conn:
        tables = [
            "dbgen_version",
            "customer_address", "customer_demographics", "date_dim", "warehouse", "ship_mode", "time_dim",
            "reason", "income_band", "item", "store", "call_center", "customer", "web_site", "store_returns",
            "household_demographics", "web_page", "promotion", "catalog_page", "inventory", "catalog_returns",
            "web_returns", "web_sales", "catalog_sales", "store_sales",
        ]
        for table in tables:
            _exec(conn, f"DROP TABLE IF EXISTS {table} CASCADE")
        with open(Path("/dsb_schema") / "create_tables.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
        for table in tables:
            table_path = Config.DSB_DATA / f"{table}.dat"
            _exec(conn, f"COPY {table} FROM '{str(table_path)}' CSV DELIMITER '|'")
        with open(Path("/dsb_schema") / "dsb_index_pg.sql") as f:
            contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
            for sql in pglast.split(contents):
                _exec(conn, sql)
    engine.dispose()


def prewarm_all(gym_config: GymConfig):
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
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


def vacuum_analyze_all(gym_config: GymConfig):
    engine = create_engine(
        gym_config.db_config.get_uri(), poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    inspector: Inspector = inspect(engine)
    with engine.connect() as conn:
        for table in inspector.get_table_names():
            # _exec(conn, f"VACUUM FULL ANALYZE {table}")
            _exec(conn, f"VACUUM ANALYZE {table}")
    engine.dispose()


def prepare(gym_config: GymConfig):
    # TODO(WAN):
    #  I'm back in an OLAP world, so I only need to run this once at the start of the gym.
    #  Otherwise, I need to bring back:
    #  - PREPARE for high throughput OLTP.
    #  - Reloading state between gym iterations, arguably you want to push this logic down further.
    #  Code that does this exists as of time of writing (see meow or nyoom branch), I just hate it.
    vacuum_analyze_all(gym_config)
    prewarm_all(gym_config)


def create_gym_configs_dsb(db_config: DbConfig) -> list[GymConfig]:
    result = []

    train_seed = 15721
    test_seed = 15722
    seed = 15721
    should_overwrite = False

    gym_configs = [
        GymConfig(
            db_config=db_config,
            expt_name="dsb_test",
            workload=[WorkloadDSB("default", test_seed)],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="dsb_default",
            workload=[WorkloadDSB("default", train_seed)],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="dsb_naive_sql",
            workload=[WorkloadDSB("naive", train_seed)],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="dsb_smart_sql",
            workload=[WorkloadDSB("smart", train_seed)],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="dsb_tskip_sample",
            workload=[WorkloadDSB("default", train_seed)],
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
            db_config=db_config,
            expt_name="dsb_tskip_cutoff",
            workload=[WorkloadDSB("default", train_seed)],
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
            db_config=db_config,
            expt_name="dsb_tskip_sample_cutoff",
            workload=[WorkloadDSB("default", train_seed)],
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
        result.append(gym_config)

    return result


def create_gym_configs_tpch(db_config: DbConfig) -> list[GymConfig]:
    result = []

    # Setup TPC-H worklaads.
    workload_seed_start, workload_seed_end = Config.TPCH_WORKLOAD_SEED_START, Config.TPCH_WORKLOAD_SEED_END
    workloads = [WorkloadTPCH(seed) for seed in range(workload_seed_start, workload_seed_end + 1)]
    default_workload, test_workload = train_test_split(workloads, test_size=0.2)
    seed = 15721
    should_overwrite = False

    gym_configs = [
        GymConfig(
            db_config=db_config,
            expt_name="tpch_test",
            workload=test_workload,
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="tpch_default",
            workload=default_workload,
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="tpch_naive_sql",
            workload=[WorkloadNaiveTablesampleTPCH(workload) for workload in default_workload],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="tpch_smart_sql",
            workload=[WorkloadSmartTablesampleTPCH(workload) for workload in default_workload],
            seed=seed,
            should_nyoom=False,
            should_overwrite=should_overwrite,
            setup_sql=[],
            nyoom_args=None,
        ),
        GymConfig(
            db_config=db_config,
            expt_name="tpch_tskip_sample",
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
            db_config=db_config,
            expt_name="tpch_tskip_cutoff",
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
            db_config=db_config,
            expt_name="tpch_tskip_sample_cutoff",
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
        result.append(gym_config)

    return result


def run_gym_config(gym_config: GymConfig):
    db_snapshot_path = Path(f"/dbgym/{gym_config.db_config.db_name}_snapshot.pkl").absolute()

    if not gym_config.should_run():
        print(f"Skipping: {gym_config.expt_name}")
        return

    print(f"Running: {gym_config.expt_name}")
    # TODO(WAN): previously we ran the gym multiple times for each initialization of the database, but vacuum etc.
    assert isinstance(gym_config.db_config, PgConfig)
    with PostgresTrainer(
            service_url=Config.TRAINER_URL, pg_config=gym_config.db_config, force_rebuild=False
    ) as trainer:
        assert trainer.dbms_db_exists(), "Startup failed?"
        trainer.dbms_install_nyoom()
        pgtune(gym_config)
        trainer.dbms_restart()
        newly_loaded = load_if_not_exists(gym_config)
        prepare(gym_config)

        # TODO(WAN): assumes read-only.
        if newly_loaded or not db_snapshot_path.exists():
            print("Snapshot: generating.")
            engine = create_engine(
                gym_config.db_config.get_uri(),
                poolclass=NullPool,
                execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            db_snapshot = DatabaseSnapshot(engine)
            engine.dispose()
            db_snapshot.to_file(db_snapshot_path)
            print("Snapshot: complete.")
        db_snapshot = DatabaseSnapshot.from_file(db_snapshot_path)

        gym_config.force_run(db_snapshot)


def main():
    db_port = int(os.getenv("TRAINER_PG_PORT"))

    db_name = f"dsb_sf{Config.DSB_SF}"
    db_config = PgConfig(
        hostname="trainer", db_user="trainer_user", db_pass="trainer_pass", db_port=db_port,
        dataset="dsb", db_name=db_name,
    )
    for gym_config in create_gym_configs_dsb(db_config):
        run_gym_config(gym_config)

    db_name = f"tpch_sf{Config.TPCH_SF}"
    db_config = PgConfig(
        hostname="trainer", db_user="trainer_user", db_pass="trainer_pass", db_port=db_port,
        dataset="tpch", db_name=db_name,
    )
    for gym_config in create_gym_configs_tpch(db_config):
        run_gym_config(gym_config)

    # test_df_name = "tpch_test"
    # train_df_names = [
    #     "tpch_default",
    #     "tpch_naive_sql",
    #     "tpch_smart_sql",
    #     "tpch_tskip_sample",
    #     "tpch_tskip_cutoff",
    #     "tpch_tskip_sample_cutoff",
    # ]
    #
    # GymConfigModel.generate_model(test_df_name=test_df_name, train_df_names=train_df_names)
    # GymConfigPlot.generate_plot(
    #     plot_suffix="tpch",
    #     expt_names=train_df_names,
    #     plot_names=train_df_names,
    # )
    # GymConfigPlot.generate_plot_runtime_by_operator(
    #     plot_suffix="tpch",
    #     expt_names=train_df_names,
    #     plot_names=train_df_names,
    # )


if __name__ == "__main__":
    main()
