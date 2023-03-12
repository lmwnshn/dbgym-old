# TODO(WAN):
#  Bring back all the scheduling and PREPARE stuff that made sense in an OLTP world.
#  Bring back the [historical, future] workload split if we're trying to do forecasting.
import copy
import os
import time
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pglast
import requests
from autogluon.tabular import TabularDataset, TabularPredictor
from dbgym.config import Config
from dbgym.env.dbgym import DbGymEnv
from dbgym.space.action.fake_index import FakeIndexSpace
from dbgym.space.observation.qppnet.features import QPPNetFeatures
from dbgym.state.database_snapshot import DatabaseSnapshot
from dbgym.trainer.postgres import PostgresTrainer
from dbgym.workload.workload import Workload, WorkloadTPCH
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
        pass
    else:
        load(dataset)


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


def gym(name, db_snapshot_path, workloads, seed=15721, overwrite=True):
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
            setup_sqls=["create extension if not exists nyoom"],
        )

        observation, info = env.reset(seed=15721)
        df = observation_space.convert_observations_to_df(observation)

        pd.Series({"Runtime (s)": df["Actual Total Time (us)"].sum() / 1e6}).to_pickle(
            Config.SAVE_PATH_OBSERVATION / name / "runtime.pkl"
        )

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


class AutogluonModel:
    def __init__(self, save_path: Path):
        self.save_path: Path = save_path
        self.predictor: TabularPredictor = TabularPredictor(label="Actual Total Time (us)", path=str(self.save_path))

    def try_load(self) -> bool:
        try:
            self.predictor.load(str(self.save_path))
            # TODO(WAN): leak of autogluon impl details, but sometimes fit doesn't save?
            return self.predictor._learner.is_fit
        except FileNotFoundError:
            return False

    def train(self, dataset: TabularDataset, time_limit=Config.AUTOGLUON_TIME_LIMIT_S):
        self.predictor.fit(dataset, time_limit=time_limit)
        pd.Series({"Training Time (s)": time_limit}).to_pickle(self.save_path / "training_time.pkl")
        self.predictor.save()

    def eval(self, dataset: TabularDataset) -> pd.DataFrame:
        y_pred = self.predictor.predict(dataset)
        eval_df = pd.concat([y_pred, dataset["Actual Total Time (us)"]], axis=1)
        eval_df.columns = ["Predicted Latency (us)", "Actual Latency (us)"]
        eval_df["diff (us)"] = (eval_df["Predicted Latency (us)"] - eval_df["Actual Latency (us)"]).abs()
        eval_df["q_err"] = np.nan_to_num(
            np.maximum(
                eval_df["Predicted Latency (us)"] / eval_df["Actual Latency (us)"],
                eval_df["Actual Latency (us)"] / eval_df["Predicted Latency (us)"],
            ),
            nan=np.inf,
        )
        return eval_df

    @staticmethod
    def make_padded_datasets(dfs: list[pd.DataFrame]) -> list[TabularDataset]:
        result = []
        max_lens = {}
        for df in dfs:
            for col in ["Children Observation Indexes", "Features", "Query Hash"]:
                max_lens[col] = max([len(x) for x in df[col].tolist()])
        for df in dfs:
            flatteneds = []
            for col in ["Children Observation Indexes", "Features", "Query Hash"]:
                col_df = pd.DataFrame(df[col].tolist(), index=df.index)
                col_df = col_df.rename(columns=lambda num: f"{col}_{num}")
                for i in range(len(col_df.columns), max_lens[col] + 1):
                    col_df[f"{col}_{i}"] = pd.NA
                flatteneds.append(col_df)
            for col in ["Node Type", "Observation Index", "Query Num", "Actual Total Time (us)"]:
                flatteneds.append(df[col])
            result.append(TabularDataset(pd.concat(flatteneds, axis=1)))
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
        load_if_not_exists("tpch")
        prepare()

        # TODO(WAN): assumes read-only.
        db_snapshot_path = Path("/dbgym/snapshot.pkl").absolute()
        if not db_snapshot_path.exists():
            print("Snapshot: generating.")
            engine = create_engine(
                Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
            )
            db_snapshot = DatabaseSnapshot(engine)
            engine.dispose()
            db_snapshot.to_file(db_snapshot_path)
            print("Snapshot: complete.")

        gym("test", db_snapshot_path, test_workloads, seed=seed, overwrite=False)
        gym("default", db_snapshot_path, default_workloads, seed=seed, overwrite=False)
        gym("tablesample", db_snapshot_path, tablesample_workloads, seed=seed, overwrite=False)

        engine = create_engine(
            Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
        )
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS nyoom"))
        engine.dispose()

        req = requests.post(Config.NYOOM_URL + "/nyoom/start/")
        assert req.status_code == 200
        print("nyoom_start: ", req.text)
        gym("default_with_nyoom", db_snapshot_path, default_workloads, seed=seed, overwrite=True)
        req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
        assert req.status_code == 200
        print("nyoom_stop: ", req.text)


def generate_model():
    test_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "test" / "0.parquet")
    default_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default" / "0.parquet")
    tablesample_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "tablesample" / "0.parquet")
    default_with_nyoom_df = pd.read_parquet(Config.SAVE_PATH_OBSERVATION / "default_with_nyoom" / "0.parquet")

    data_dfs = []
    data_dfs.append(test_df)
    data_dfs.append(default_df)
    data_dfs.append(tablesample_df)
    data_dfs.append(default_with_nyoom_df)
    for i, df in enumerate(data_dfs):
        print("Data", i, df.shape)

    autogluon_dfs = AutogluonModel.make_padded_datasets(data_dfs)
    test_data = autogluon_dfs[0]
    default_data = autogluon_dfs[1]
    tablesample_data = autogluon_dfs[2]
    default_with_nyoom_data = autogluon_dfs[3]

    def save_model_eval(_expt_name: str, _df: pd.DataFrame, _train_data: TabularDataset, _test_data: TabularDataset):
        if (Config.SAVE_PATH_EVAL / _expt_name).exists():
            return

        expt_autogluon = AutogluonModel(Config.SAVE_PATH_MODEL / _expt_name)
        if not expt_autogluon.try_load():
            expt_autogluon.train(_train_data)

        (Config.SAVE_PATH_EVAL / _expt_name).mkdir(parents=True, exist_ok=True)
        accuracy = expt_autogluon.eval(_test_data)
        accuracy.to_parquet(Config.SAVE_PATH_EVAL / _expt_name / "accuracy.parquet")

    save_model_eval("default", default_df, default_data, test_data)
    save_model_eval("tablesample", tablesample_df, tablesample_data, test_data)
    save_model_eval("default_with_nyoom", default_with_nyoom_df, default_with_nyoom_data, test_data)


def generate_plot():
    def load_model_eval(_expt_name: str) -> pd.DataFrame:
        return pd.read_parquet(Config.SAVE_PATH_EVAL / _expt_name)

    def read_runtime(_expt_name: str) -> float:
        return pd.read_pickle(Config.SAVE_PATH_OBSERVATION / _expt_name / "runtime.pkl")["Runtime (s)"]

    def read_training_time(_expt_name: str) -> float:
        return pd.read_pickle(Config.SAVE_PATH_MODEL / _expt_name / "training_time.pkl")["Training Time (s)"]

    labeled_expt = [
        # (code name, plot name)
        ("default", "Default"),
        ("tablesample", "Sample"),
        (None, "VerdictDB"),
        (None, "QPE"),
        (None, "TSkip"),
        ("default_with_nyoom", "Nyoom"),
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
            metrics = load_model_eval(expt_name)
            mae_s.append(metrics["diff (us)"].mean() / 1e6)
            runtime_s.append(read_runtime(expt_name))
            training_time_s.append(read_training_time(expt_name))
        index.append(index_name)

    Config.SAVE_PATH_PLOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    df = pd.DataFrame({"MAE (s)": mae_s}, index=index)
    df.plot.bar(ax=ax, rot=45)
    fig.savefig(Config.SAVE_PATH_PLOT / f"accuracy.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    df = pd.DataFrame(
        {
            "Runtime (s)": runtime_s,
            "Training Time (s)": training_time_s,
        },
        index=index,
    )
    df.plot.bar(stacked=True, ax=ax, rot=45)
    fig.savefig(Config.SAVE_PATH_PLOT / f"runtime.pdf")
    plt.close(fig)


def main():
    generate_data()
    generate_model()
    generate_plot()


if __name__ == "__main__":
    main()
