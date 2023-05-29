# TODO(WAN):
#  Bring back all the scheduling and PREPARE stuff that made sense in an OLTP world.
#  Bring back the [historical, future] workload split if we're trying to do forecasting.
import copy
import os
import time
from pathlib import Path

import gymnasium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pglast
import requests
from autogluon.tabular import TabularDataset
from dbgym.config import Config
from dbgym.env.dbgym import DbGymEnv
from dbgym.model.autogluon import AutogluonModel
from dbgym.space.action.fake_index import FakeIndexSpace
from dbgym.space.observation.qppnet.features import QPPNetFeatures
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.trainer.postgres import PostgresTrainer
from dbgym.workload.workload import Workload, WorkloadTPCH
from sklearn.model_selection import train_test_split
from sqlalchemy import NullPool, create_engine, inspect, text
from sqlalchemy.engine.base import Connection
from sqlalchemy.engine.reflection import Inspector

plt.rcParams["axes.grid"] = True
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["figure.autolayout"] = True
plt.rcParams["grid.color"] = "silver"
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["image.cmap"] = "tab10"
plt.rcParams["legend.frameon"] = False
plt.rcParams["savefig.bbox"] = "tight"
# If 0, the axes spines are eaten. https://github.com/matplotlib/matplotlib/issues/7806
# Use pdfcrop to fix post-render.
plt.rcParams["savefig.pad_inches"] = 0.05

# From Matt, figsize is (3.5,2) for half-page and (7,2) for full-page.
figsize_full = (7.0, 2.0)
figsize_half = (3.5, 2.0)
figsize_quarter = (1.75, 1.0)
fig_dpi = 600
font_mini, font_tiny, font_small, font_medium, font_large, font_huge = 4, 6, 8, 10, 12, 14

plt.rcParams["font.family"] = "Liberation Sans"
# matplotlib defaults to Type 3 fonts, which are full PostScript fonts.
# Some publishers only accept Type 42 fonts, which are PostScript wrappers around TrueType fonts.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files


def fig_quarter():
    plt.rcParams["figure.figsize"] = figsize_quarter
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["font.size"] = font_mini
    plt.rcParams["axes.titlesize"] = font_mini
    plt.rcParams["axes.labelsize"] = font_tiny
    plt.rcParams["xtick.labelsize"] = font_mini
    plt.rcParams["ytick.labelsize"] = font_mini
    plt.rcParams["legend.fontsize"] = font_mini
    plt.rcParams["figure.titlesize"] = font_small

    for var in ["xtick", "ytick"]:
        plt.rcParams[f"{var}.major.size"] = 3.5 / 2
        plt.rcParams[f"{var}.minor.size"] = 2 / 2
        plt.rcParams[f"{var}.major.width"] = 0.8 / 2
        plt.rcParams[f"{var}.minor.width"] = 0.6 / 2
        plt.rcParams[f"{var}.major.pad"] = 3.5 / 2
        plt.rcParams[f"{var}.minor.pad"] = 3.4 / 2
    plt.rcParams["axes.linewidth"] = 0.8 / 2
    plt.rcParams["grid.linewidth"] = 0.8 / 2
    plt.rcParams["lines.linewidth"] = 2 / 2
    plt.rcParams["lines.markersize"] = 6 / 2


def fig_half():
    plt.rcParams["figure.figsize"] = figsize_half
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["font.size"] = font_small
    plt.rcParams["axes.titlesize"] = font_small
    plt.rcParams["axes.labelsize"] = font_medium
    plt.rcParams["xtick.labelsize"] = font_small
    plt.rcParams["ytick.labelsize"] = font_small
    plt.rcParams["legend.fontsize"] = font_small
    plt.rcParams["figure.titlesize"] = font_large

    for var in ["xtick", "ytick"]:
        plt.rcParams[f"{var}.major.size"] = 3.5
        plt.rcParams[f"{var}.minor.size"] = 2
        plt.rcParams[f"{var}.major.width"] = 0.8
        plt.rcParams[f"{var}.minor.width"] = 0.6
        plt.rcParams[f"{var}.major.pad"] = 3.5
        plt.rcParams[f"{var}.minor.pad"] = 3.4
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.linewidth"] = 0.8
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 6


def fig_full():
    plt.rcParams["figure.figsize"] = figsize_full
    plt.rcParams["figure.dpi"] = fig_dpi
    plt.rcParams["font.size"] = font_medium
    plt.rcParams["axes.titlesize"] = font_medium
    plt.rcParams["axes.labelsize"] = font_large
    plt.rcParams["xtick.labelsize"] = font_medium
    plt.rcParams["ytick.labelsize"] = font_medium
    plt.rcParams["legend.fontsize"] = font_medium
    plt.rcParams["figure.titlesize"] = font_huge

    # TODO(WAN): yet to make a full figure.


nyoom_configs = [
    # {"method": "optimizer", "optimizer_cutoff_pct": 10, "optimizer_min_processed": 0},
    # {"method": "optimizer", "optimizer_cutoff_pct": 10, "optimizer_min_processed": 1000},
    {"method": "optimizer", "optimizer_cutoff_pct": 20, "optimizer_min_processed": 0},
    # {"method": "optimizer", "optimizer_cutoff_pct": 20, "optimizer_min_processed": 1000},
    # {"method": "optimizer", "optimizer_cutoff_pct": 50, "optimizer_min_processed": 0},
    # {"method": "optimizer", "optimizer_cutoff_pct": 50, "optimizer_min_processed": 1000},
    # {"method": "tskip", "tskip_wiggle_std": 1.0, "tskip_wiggle_sampen": 20},
    # {"method": "tskip", "tskip_wiggle_std": 1.5, "tskip_wiggle_sampen": 20},
    # {"method": "tskip", "tskip_wiggle_std": 2.0, "tskip_wiggle_sampen": 20},
    {"method": "tskip", "tskip_wiggle_std": 2.5, "tskip_wiggle_sampen": 20},
    # {"method": "tskip", "tskip_wiggle_std": 3.0, "tskip_wiggle_sampen": 20},
    # {"method": "tskip", "tskip_wiggle_std": 2.5, "tskip_wiggle_sampen": 50},
    # {"method": "tskip", "tskip_wiggle_std": 3.0, "tskip_wiggle_sampen": 50},
]
tws_ttc = [
    # (10000, 10000),
    (10000, 25000),
    # (10000, 50000),
    # (10000, 75000),
    # (10000, 100000),
]


def get_experiment_name(tws, ttc, nc):
    name = f"nyoom_tws_{tws}_ttc_{ttc}_method_{nc['method']}"
    if nc["method"] == "tskip":
        name = name + f"_std_{nc['tskip_wiggle_std']}_sampen_{nc['tskip_wiggle_sampen']}"
    elif nc["method"] == "optimizer":
        name = name + f"_cutoff_{nc['optimizer_cutoff_pct']}_mp_{nc['optimizer_min_processed']}"
    return name


def get_experiment_names():
    names = []
    for (tws, ttc) in tws_ttc:
        for nc in nyoom_configs:
            names.append(get_experiment_name(tws, ttc, nc))
    return names


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


def gym(name, db_snapshot_path, workloads, setup_sqls=None, seed=15721, overwrite=True):
    if setup_sqls is None:
        setup_sqls = []
    # TODO(WAN): HACK!!
    setup_sqls.append("set max_parallel_workers_per_gather = 0")
    db_snapshot = DatabaseSnapshot.from_file(db_snapshot_path)

    # Run the queries in the gym.
    obs_path = Config.SAVE_PATH_OBSERVATION / name
    obs_path.mkdir(parents=True, exist_ok=True)
    obs_iter = 0
    pq_path = obs_path / f"{obs_iter}.parquet"

    if overwrite or not pq_path.exists():
        action_space = FakeIndexSpace(1)
        observation_space = QPPNetFeatures(db_snapshot=db_snapshot, seed=seed)

        # noinspection PyTypeChecker
        env: DbGymEnv = gymnasium.make(
            "dbgym/DbGym-v0",
            disable_env_checker=True,
            name=name,
            action_space=action_space,
            observation_space=observation_space,
            connstr=Config.TRAINER_PG_URI,
            workloads=workloads,
            seed=seed,
            setup_sqls=setup_sqls,
        )

        observation, info = env.reset(seed=15721)
        df = observation_space.convert_observations_to_df(observation)

        df.to_parquet(pq_path)
        obs_iter += 1

        # TODO(WAN): eventually, tuning. Will need to change the pq_path.exists() check above.
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


def hack_tablesample_tpch(workload: WorkloadTPCH) -> Workload:
    result = copy.deepcopy(workload)

    # TODO(WAN):
    #  I really hate this code. TABLESAMPLE only at root-level to try to limit the perf impact caused by PostgreSQL
    #  having a poor optimizer.

    # TODO(WAN): try this
    # sample_method = "BERNOULLI (1)"

    sample_method = "BERNOULLI (10)"
    sample_seed = "REPEATABLE (15721)"
    for i, query in enumerate(result.queries, 1):
        if i == 1:
            query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 2:
            query = query.replace("\n\tpart,", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 3:
            query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 4:
            query = query.replace("\n\torders", f"\n\torders TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 5:
            query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 6:
            query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 7:
            query = query.replace("\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tnation n2", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 8:
            query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tnation n2,", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tregion", f"\n\t\t\tregion TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 9:
            query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tpartsupp,", f"\n\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\t\t\tnation", f"\n\t\t\tnation TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 10:
            query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 11:
            query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 12:
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 13:
            query = query.replace(
                "\n\t\t\tcustomer left outer join orders",
                f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed} left outer join orders TABLESAMPLE {sample_method} {sample_seed}",
            )
        elif i == 14:
            query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 15:
            query = query.replace("\n\t\tlineitem", f"\n\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 15 + 1:
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
        elif i == 15 + 2:
            pass
        elif i == 16 + 2:
            query = query.replace("\n\tpartsupp,", f"\n\tPARTSUPP TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            query = query.replace("PARTSUPP", "partsupp")
        elif i == 17 + 2:
            query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 18 + 2:
            query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 19 + 2:
            query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 20 + 2:
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 21 + 2:
            query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tlineitem l1,", f"\n\tlineitem l1 TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
            query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
        elif i == 22 + 2:
            query = query.replace("\n\t\t\tcustomer", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed}")
        result.queries[i - 1] = query
    return result


def generate_data():
    workload_seed_start, workload_seed_end = Config.WORKLOAD_SEED_START, Config.WORKLOAD_SEED_END
    workloads = [WorkloadTPCH(seed) for seed in range(workload_seed_start, workload_seed_end + 1)]
    default_workloads, test_workloads = train_test_split(workloads, test_size=0.2)
    tablesample_workloads = [hack_tablesample_tpch(workload) for workload in default_workloads]

    seed = 15721
    with PostgresTrainer(Config.TRAINER_URL, force_rebuild=False) as trainer:
        assert trainer.dbms_exists(), "Startup failed?"
        trainer.dbms_install_nyoom()
        pgtune()
        trainer.dbms_restart()
        loaded = load_if_not_exists("tpch")
        prepare()

        # TODO(WAN): assumes read-only.
        db_snapshot_path = Path("/dbgym/snapshot.pkl").absolute()
        if loaded or not db_snapshot_path.exists():
            print("Snapshot: generating.")
            engine = create_engine(
                Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            db_snapshot = DatabaseSnapshot(engine)
            engine.dispose()
            db_snapshot.to_file(db_snapshot_path)
            print("Snapshot: complete.")

        engine = create_engine(
            Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
        )
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS nyoom"))
        engine.dispose()

        setup_sqls = ["CREATE EXTENSION IF NOT EXISTS nyoom"]
        gym("test", db_snapshot_path, test_workloads, seed=seed, setup_sqls=setup_sqls, overwrite=False)
        gym("default", db_snapshot_path, default_workloads, seed=seed, setup_sqls=setup_sqls, overwrite=False)
        gym("tablesample", db_snapshot_path, tablesample_workloads, seed=seed, setup_sqls=setup_sqls, overwrite=False)

        for (tws, ttc) in tws_ttc:
            for nc in nyoom_configs:
                name = get_experiment_name(tws, ttc, nc)
                nyoom_overwrite = False

                # TODO(WAN): hack to prevent sending superfluous stop/start messages
                obs_path = Config.SAVE_PATH_OBSERVATION / name
                obs_path.mkdir(parents=True, exist_ok=True)
                obs_iter = 0
                pq_path = obs_path / f"{obs_iter}.parquet"

                gym_will_run = nyoom_overwrite or not pq_path.exists()
                if not gym_will_run:
                    print(f"Skipping: {name}")

                    default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")
                    default_exec_time = default_df.groupby(["Query Num"]).first()["Execution Time (ms)"].sum()

                    nyoom_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / name / "0.parquet")
                    nyoom_exec_time = nyoom_df.groupby(["Query Num"]).first()["Execution Time (ms)"].sum()

                    if nyoom_exec_time > 0.8 * default_exec_time:
                        # TODO(WAN): Sometimes it bugs out, so we'll just rerun. It seems to fortunately only
                        #            bug out at the level of individual experiments, which is probably related to
                        #            the earlier hammering with start and stop.
                        print(f"Rerunning: {name}")
                        nyoom_overwrite = True
                    else:
                        continue
                print(f"Running: {name}")

                req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
                # TODO(WAN): pixie dust
                time.sleep(5)
                req = requests.post(Config.NYOOM_URL + "/nyoom/start/")
                assert req.status_code == 200
                setup_sqls = [
                    "CREATE EXTENSION IF NOT EXISTS nyoom",
                    f"SET nyoom.telemetry_window_size = {tws}",
                    f"SET nyoom.telemetry_tuple_count = {ttc}",
                    # # TODO(WAN): parallel tests
                    # f"SET max_parallel_workers_per_gather = 0",
                ]
                print("nyoom_start: ", req.text)
                gym(name, db_snapshot_path, default_workloads, setup_sqls=setup_sqls, seed=seed, overwrite=nyoom_overwrite)
                req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
                assert req.status_code == 200
                print("nyoom_stop: ", req.text)


def generate_seqscan_data():
    print("generate_seqscan_data")
    telemetry_window_size = 100000
    telemetry_tuple_counts = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    workloads = [Workload(["SELECT * FROM lineitem"]) for _ in range(100)]
    default_workloads, test_workloads = train_test_split(workloads, test_size=0.2)

    configs = [
        (default_workloads, telemetry_window_size, ttc)
        for ttc in telemetry_tuple_counts
    ]

    seed = 15721
    with PostgresTrainer(Config.TRAINER_URL, force_rebuild=False) as trainer:
        assert trainer.dbms_exists(), "Startup failed?"
        trainer.dbms_install_nyoom()
        pgtune()
        trainer.dbms_restart()
        loaded = load_if_not_exists("tpch")
        prepare()

        # TODO(WAN): assumes read-only.
        db_snapshot_path = Path("/dbgym/snapshot.pkl").absolute()
        if loaded or not db_snapshot_path.exists():
            print("Snapshot: generating.")
            engine = create_engine(
                Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            db_snapshot = DatabaseSnapshot(engine)
            engine.dispose()
            db_snapshot.to_file(db_snapshot_path)
            print("Snapshot: complete.")

        gym("test", db_snapshot_path, test_workloads, seed=seed, overwrite=False)

        seed = 15721
        for workload, tws, ttc in configs:
            setup_sqls = [
                "CREATE EXTENSION IF NOT EXISTS nyoom",
                f"SET nyoom.telemetry_window_size = {tws}",
                f"SET nyoom.telemetry_tuple_count = {ttc}",
            ]

            expt_name = f"seqscan_lineitem_tws{tws}_ttc{ttc}"
            gym(expt_name, db_snapshot_path, workload, setup_sqls=setup_sqls, seed=seed, overwrite=False)

            engine = create_engine(
                Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS nyoom"))
            engine.dispose()

            req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
            req = requests.post(Config.NYOOM_URL + "/nyoom/start/")
            assert req.status_code == 200
            print("nyoom_start: ", req.text)
            expt_name = f"seqscan_lineitem_tws{tws}_ttc{ttc}_nyoom"
            gym(expt_name, db_snapshot_path, workload, setup_sqls=setup_sqls, seed=seed, overwrite=False)
            req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
            assert req.status_code == 200
            print("nyoom_stop: ", req.text)


class Model:
    @staticmethod
    def save_model_eval(_expt_name: str, _df: pd.DataFrame, _train_data: TabularDataset, _test_data: TabularDataset,
                        predictor_target: str):
        eval_exists = (Config.SAVE_PATH_EVAL / _expt_name).exists()
        eval_newer = eval_exists and os.path.getmtime(Config.SAVE_PATH_EVAL / _expt_name) > os.path.getmtime(Config.SAVE_PATH_OBSERVATION / _expt_name / "0.parquet")

        if eval_exists and eval_newer:
            return

        expt_autogluon = AutogluonModel(Config.SAVE_PATH_MODEL / _expt_name, predictor_target=predictor_target)
        if not expt_autogluon.try_load():
            expt_autogluon.train(_train_data)

        (Config.SAVE_PATH_EVAL / _expt_name).mkdir(parents=True, exist_ok=True)
        accuracy = expt_autogluon.eval(_test_data)
        accuracy.to_parquet(Config.SAVE_PATH_EVAL / _expt_name / "accuracy.parquet")

    @staticmethod
    def generate_model():
        test_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "test" / "0.parquet")
        default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")
        tablesample_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "tablesample" / "0.parquet")

        names = get_experiment_names()
        nyoom_dfs = []
        for name in names:
            nyoom_dfs.append(pd.read_parquet(Config.SAVE_PATH_OBSERVATION / name / "0.parquet"))

        # tablesample_hack_df = tablesample_df.copy()
        # tablesample_hack_df["Node Type"] = tablesample_hack_df["Node Type"].replace({"Sample Scan": "Seq Scan"})
        # Model.save_model_eval("tablesample_hack", tablesample_hack_df, tablesample_hack_data, test_data, predictor_target=predictor_target)

        data_dfs = [
            test_df,
            default_df,
            tablesample_df,
            *nyoom_dfs,
        ]
        for i, df in enumerate(data_dfs):
            print("Data", i, df.shape)

        predictor_target = "Actual Total Time (ms)"
        autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs, predictor_target=predictor_target)
        (test_data,
         default_data,
         tablesample_data,
         *nyoom_datas) \
            = autogluon_dfs

        Model.save_model_eval("default", default_df, default_data, test_data, predictor_target=predictor_target)
        Model.save_model_eval("tablesample", tablesample_df, tablesample_data, test_data,
                              predictor_target=predictor_target)
        for name, (nyoom_df, nyoom_data) in zip(names, zip(nyoom_dfs, nyoom_datas)):
            Model.save_model_eval(name, nyoom_df, nyoom_data, test_data, predictor_target=predictor_target)

    @staticmethod
    def generate_model_noise_tpch():
        test_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "test" / "0.parquet")
        default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")

        rng = np.random.default_rng(15721)

        under_df = default_df.copy()
        under_df["Actual Total Time (ms)"] = under_df["Actual Total Time (ms)"] * rng.uniform(0.5, 1, under_df.shape[0])

        over_df = default_df.copy()
        over_df["Actual Total Time (ms)"] = over_df["Actual Total Time (ms)"] * rng.uniform(1, 1.5, over_df.shape[0])

        gaussian_df = default_df.copy()
        gaussian_df["Actual Total Time (ms)"] = gaussian_df["Actual Total Time (ms)"].apply(
            lambda x: max(0, rng.normal(loc=x, scale=0.33 * x))
        )

        data_dfs = []
        data_dfs.append(test_df)
        data_dfs.append(default_df)
        data_dfs.append(under_df)
        data_dfs.append(over_df)
        data_dfs.append(gaussian_df)
        for i, df in enumerate(data_dfs):
            print("Data", i, df.shape)

        predictor_target = "Actual Total Time (ms)"
        autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs, predictor_target=predictor_target)
        test_data = autogluon_dfs[0]
        default_data = autogluon_dfs[1]
        under_data = autogluon_dfs[2]
        over_data = autogluon_dfs[3]
        gaussian_data = autogluon_dfs[4]
        Model.save_model_eval(f"default", default_df, default_data, test_data, predictor_target=predictor_target)
        Model.save_model_eval(f"default_under", under_df, under_data, test_data, predictor_target=predictor_target)
        Model.save_model_eval(f"default_over", over_df, over_data, test_data, predictor_target=predictor_target)
        Model.save_model_eval(f"default_gaussian", gaussian_df, gaussian_data, test_data,
                              predictor_target=predictor_target)

    @staticmethod
    def generate_model_sweep_tpch():
        test_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "test" / "0.parquet")
        default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")

        data_dfs = []
        data_dfs.append(test_df)
        data_dfs.append(default_df)
        for i, df in enumerate(data_dfs):
            print("Data", i, df.shape)

        predictor_target = "Actual Total Time (ms)"
        autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs)
        test_data = autogluon_dfs[0]
        default_data = autogluon_dfs[1]
        Model.save_model_eval(f"default", default_df, default_data, test_data, predictor_target=predictor_target)

        for pct in reversed(list(range(10, 90 + 1, 10))):
            pct_num_rows = int(default_df.shape[0] * pct / 100)
            pct_df = default_df.head(pct_num_rows)

            data_dfs = []
            data_dfs.append(test_df)
            data_dfs.append(pct_df)
            for i, df in enumerate(data_dfs):
                print("Data", i, df.shape)

            autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs, predictor_target=predictor_target)
            test_data = autogluon_dfs[0]
            pct_data = autogluon_dfs[1]
            Model.save_model_eval(f"pct_{pct}", pct_df, pct_data, test_data, predictor_target=predictor_target)


class Plot:
    @staticmethod
    def load_model_eval(_expt_name: str) -> pd.DataFrame:
        return pd.read_parquet(Config.SAVE_PATH_EVAL / _expt_name)

    @staticmethod
    def read_runtime(_expt_name: str) -> float:
        df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / _expt_name / "0.parquet")
        return df.groupby(["Query Num"]).first()["Execution Time (ms)"].sum()

    @staticmethod
    def read_training_time(_expt_name: str) -> float:
        return pd.read_pickle(Config.SAVE_PATH_MODEL / _expt_name / "training_time.pkl")["Training Time (s)"]

    @staticmethod
    def generate_plot():
        names = get_experiment_names()

        for name in names:
            labeled_expt = [
                # (code name, plot name)
                ("default", "Default"),
                ("tablesample", "Sample"),
                # ("tablesample_hack", "SampleHack"),
                # ("default_with_nyoom", "TSkip"),
                (name, name),
            ]

            mae_s = []
            runtime_s = []
            training_time_s = []
            index = []
            for expt_name, index_name in labeled_expt:
                if expt_name is None:
                    # TODO(WAN): temporary hack until we get those working.
                    mae_s.append(0)
                    runtime_s.append(0)
                    training_time_s.append(0)
                else:
                    metrics = Plot.load_model_eval(expt_name)
                    mae_s.append(metrics["Diff (ms)"].mean() / 1e3)
                    if expt_name == "tablesample_hack":
                        runtime_s.append(Plot.read_runtime("tablesample"))
                    else:
                        runtime_s.append(Plot.read_runtime(expt_name))
                    training_time_s.append(Plot.read_training_time(expt_name))
                index.append(index_name)

            Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
            fig_half()
            fig, ax = plt.subplots(1, 1)
            df = pd.DataFrame({"MAE (s)": mae_s}, index=index)
            ax = df.plot.bar(ax=ax, rot=0, legend=False)
            ax.set_ylabel("Mean Absolute Error (s)")
            fig.savefig(Config.SAVE_PATH_PLOT / f"accuracy_{name}.pdf")
            plt.close(fig)

            fig_half()
            fig, ax = plt.subplots(1, 1)
            df = pd.DataFrame(
                {
                    "Runtime (s)": runtime_s,
                    "Training Time (s)": training_time_s,
                },
                index=index,
            )
            ax = df.plot.bar(stacked=True, ax=ax, rot=0)
            ax.set_ylabel("Time (s)")
            fig.savefig(Config.SAVE_PATH_PLOT / f"runtime_{name}.pdf")
            plt.close(fig)

    @staticmethod
    def generate_plot_noise_tpch():
        labeled_expt = [
            ("default", "Baseline"),
            ("default_gaussian", "Gaussian"),
            ("default_over", "Overestimate"),
            ("default_under", "Underestimate"),
        ]

        mae_s = []
        runtime_s = []
        training_time_s = []
        index = []
        for expt_name, index_name in labeled_expt:
            metrics = Plot.load_model_eval(expt_name)
            mae_s.append(metrics["Diff (ms)"].mean() / 1e3)
            runtime_s.append(Plot.read_runtime("default"))
            training_time_s.append(Plot.read_training_time(expt_name))
            index.append(index_name)

        Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
        fig_half()
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame({"MAE (s)": mae_s}, index=index)
        ax = df.plot.bar(ax=ax, rot=0, legend=False)
        ax.set_ylabel("Mean Absolute Error (s)")
        fig.savefig(Config.SAVE_PATH_PLOT / f"noise_tpch.pdf")
        plt.close(fig)

    @staticmethod
    def generate_plot_sweep_tpch():
        pcts = list(reversed(range(10, 90 + 1, 10)))
        code_names = ["default"] + [f"pct_{pct}" for pct in pcts]
        plot_names = ["100%"] + [f"{pct}%" for pct in pcts]
        labeled_expt = zip(code_names, plot_names)

        mae_s = []
        runtime_s = []
        training_time_s = []
        index = []
        for expt_name, index_name in labeled_expt:
            metrics = Plot.load_model_eval(expt_name)
            mae_s.append(metrics["Diff (ms)"].mean() / 1e3)
            runtime_s.append(Plot.read_runtime("default"))
            training_time_s.append(Plot.read_training_time(expt_name))
            index.append(index_name)

        Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
        fig_half()
        fig, ax = plt.subplots(1, 1)
        df = pd.DataFrame({"MAE (s)": mae_s}, index=index)
        ax = df.plot.bar(ax=ax, rot=0, legend=False)
        ax.set_xlabel("Percentage of Train Dataset Used")
        ax.set_ylabel("Mean Absolute Error (s)")
        fig.savefig(Config.SAVE_PATH_PLOT / f"sweep_tpch.pdf")
        plt.close(fig)

    @staticmethod
    def generate_tpch_runtime_by_operator_HACK():
        names = get_experiment_names()
        for name in names:
            default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")
            nyoom_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / name / "0.parquet")

            default_sums = default_df.groupby("Node Type")["Differenced Time (ms)"].sum()
            nyoom_sums = nyoom_df.groupby("Node Type")["Differenced Time (ms)"].sum()

            plotter = default_sums.to_frame(name="Default").join(nyoom_sums.to_frame(name="TSkip"))
            ax = plotter.plot(kind="bar", cmap=matplotlib.colormaps["tab20"])
            ax.set_ylabel("Time (ms)")
            ax.set_xlabel("Operator Type")
            plt.tight_layout()
            Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
            plt.savefig(Config.SAVE_PATH_PLOT / f"tpch_runtime_by_operator_HACK_{name}.pdf")

    @staticmethod
    def generate_tpch_runtime_by_operator():
        names = get_experiment_names()
        for name in names:
            default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")
            tablesample_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "tablesample" / "0.parquet")
            nyoom_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / name / "0.parquet")

            default_sums = default_df.groupby("Node Type")["Nyoom Differenced Total Time (ms)"].sum()
            tablesample_sums = tablesample_df.groupby("Node Type")["Nyoom Differenced Total Time (ms)"].sum()
            nyoom_sums = nyoom_df.groupby("Node Type")["Nyoom Differenced Total Time (ms)"].sum()

            plotter = default_sums.to_frame(name="Default")\
                .join(tablesample_sums.to_frame(name="Sample"), how="outer")\
                .join(nyoom_sums.to_frame(name="TSkip"), how="outer")
            ax = plotter.plot(kind="bar", cmap=matplotlib.colormaps["tab20"])
            ax.set_ylabel("Time (ms)")
            ax.set_xlabel("Operator Type")
            plt.tight_layout()
            Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
            plt.savefig(Config.SAVE_PATH_PLOT / f"tpch_runtime_by_operator_{name}.pdf")


def main():
    pass
    generate_data()
    # Model.generate_model()
    # Plot.generate_plot()
    # Model.generate_model_sweep_tpch()
    # Plot.generate_plot_sweep_tpch()
    # Model.generate_model_noise_tpch()
    # Plot.generate_plot_noise_tpch()
    # Plot.generate_tpch_runtime_by_operator_HACK()
    Plot.generate_tpch_runtime_by_operator()
    # generate_seqscan_data()


if __name__ == "__main__":
    main()
