from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload, convert_sqls_to_postgresql_csvlog

from pathlib import Path

import gym

from dbgym.spaces.actions.fake_index import FakeIndexSpace
from dbgym.spaces.observations.qppnet.features import QPPNetFeatures
from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.workload import Workload
from dbgym.envs.state import PostgresState
from dbgym.envs.trainer import PostgresTrainer
from dbgym.envs.database_snapshot import DatabaseSnapshot

from sqlalchemy import create_engine

def convert_tpch_queries_to_workload(artifact_path):
    csvlog_path = artifact_path / "prod_dbms" / "tpch.csv"
    db_path = artifact_path / "prod_dbms" / "tpch.db"
    sqls = []
    for tpch_seed in [15721]:
        for qnum in range(1, 22+1):
            filepath = f"/home/wanshenl/git/postgres/artifact/tpch-kit/queries/{tpch_seed}/{qnum}.sql"
            with open(filepath) as f:
                lines = []
                for line in f:
                    if not line.startswith("--") and len(line.strip()) > 0:
                        lines.append(line)
                        if lines[-1].strip().endswith(";"):
                            sqls.append("".join(lines))
                            lines.clear()
                assert len(lines) == 0, f"What's here? {lines}"
    convert_sqls_to_postgresql_csvlog(sqls, csvlog_path)
    convert_postgresql_csvlog_to_workload(csvlog_path, db_path)
    return db_path


def main():
    seed = 15721
    artifact_path = Path("./artifact").absolute()
    for path in [
        artifact_path,
        artifact_path / "gym" / "observations",
        artifact_path / "gym" / "snapshots",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    state_path = artifact_path / "prod_dbms" / "state"
    tpch = Workload(convert_tpch_queries_to_workload(artifact_path), read_only=True)
    gym_spec = GymSpec(
        historical_workloads=[tpch],
        historical_state=PostgresState(state_path),
    )

    observations_path = artifact_path / "gym" / "observations" / "train.parquet"
    snapshot_path = artifact_path / "gym" / "snapshots" / "historical.pickle"
    with PostgresTrainer(
            service_url="http://localhost:5000/", gym_spec=gym_spec, seed=seed,
            gh_user="lmwnshn", gh_repo="postgres", branch="wan", build_type="release",
            db_name="noisepage_db", db_user="noisepage_user", db_pass="noisepage_pass",
            host="localhost", port=15420,
    ) as trainer:
        engine = create_engine(trainer.dbms_connstr())
        if not snapshot_path.exists():
            db_snapshot = DatabaseSnapshot(engine)
            db_snapshot.to_file(snapshot_path)
        db_snapshot = DatabaseSnapshot.from_file(snapshot_path)

        action_space = FakeIndexSpace(1)
        observation_space = QPPNetFeatures(db_snapshot=db_snapshot, seed=seed)
        runner_args = {}

        env = gym.make(
            "dbgym/DbGym-v0",
            disable_env_checker=True,
            gym_spec=gym_spec,
            trainer=trainer,
            action_space=action_space,
            observation_space=observation_space,
            runner_args=runner_args
        )

        observations, info = env.reset(seed=15721)
        train_df = observation_space.convert_observations_to_df(observations)
        train_df.to_parquet(observations_path)
        env.close()


if __name__ == "__main__":
    main()
