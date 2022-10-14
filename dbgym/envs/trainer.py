import json
from abc import ABC
from time import sleep

import psutil
from plumbum import local
from sqlalchemy import create_engine, inspect

from dbgym.envs.gym_spec import GymSpec
from dbgym.envs.state import PostgresState
from dbgym.util.plumbum_hack import PlumbumQuoteHack

createdb = local["createdb"]
createuser = local["createuser"]
dropdb = local["dropdb"]
dropuser = local["dropuser"]
pg_createcluster = local["pg_createcluster"]
pg_ctlcluster = local["pg_ctlcluster"]
pg_dropcluster = local["pg_dropcluster"]
pg_isready = local["pg_isready"]
pg_lsclusters = local["pg_lsclusters"]
pg_restore = local["pg_restore"]
psql = local["psql"]
sudo = local["sudo"]


class Trainer(ABC):
    def __init__(self, gym_spec: GymSpec, seed=15721):
        self._gym_spec: GymSpec = gym_spec
        self._seed: int = seed
        pass

    def get_target_dbms_connstr_sqlalchemy(self) -> str:
        raise NotImplementedError

    def create_target_dbms(self):
        raise NotImplementedError

    def delete_target_dbms(self):
        raise NotImplementedError


class PostgresTrainer(Trainer):
    def __init__(self, gym_spec: GymSpec, seed=15721):
        super().__init__(gym_spec, seed)
        # TODO(WAN): These should be moved into the gym specification for a PostgreSQL database.
        self._cluster_version = "14"
        self._cluster_name = "gymcluster"
        self._cluster_data = "/tmp/gym/gymdata"
        self._cluster_log = "/tmp/gym/gymlog"
        self._cluster_host = "localhost"
        self._cluster_port = 15420

        self._db_name = "gym_db"
        self._db_user = "gym_user"
        self._db_pass = "gym_pass"

        # TODO(WAN): This is a hacky way of going about things.
        if self._gym_spec.snapshot is None:
            engine = create_engine(self.get_target_dbms_connstr_sqlalchemy())
            inspector = inspect(engine)
            self._gym_spec.snapshot_db(engine, inspector)
        return
        # TODO(WAN): hack, create and delete target DBMS to populate snapshot.
        if self._gym_spec.snapshot is None:
            self.create_target_dbms()
            self.delete_target_dbms()

    def get_target_dbms_connstr_sqlalchemy(self) -> str:
        return f"postgresql+psycopg2://{self._db_user}:{self._db_pass}@{self._cluster_host}:{self._cluster_port}/{self._db_name}"

    def create_target_dbms(self):
        return
        # Setup cluster.
        if self._test_cluster_exists():
            self.delete_target_dbms()
        self._create_cluster()
        self._start_target_dbms()

        # Setup user.
        with PlumbumQuoteHack():
            drop_user_sql = f"DROP USER IF EXISTS {self._db_user}"
            create_user_sql = f"CREATE USER {self._db_user} WITH SUPERUSER ENCRYPTED PASSWORD '{self._db_pass}'"
            self._run_sql(drop_user_sql, as_postgres=True)
            self._run_sql(create_user_sql, as_postgres=True)

        # Setup DB.
        with local.env(PGPASSWORD=self._db_pass):
            dropdb[
                "--if-exists",
                "-U",
                self._db_user,
                "-h",
                self._cluster_host,
                "-p",
                self._cluster_port,
                self._db_name,
            ].run_fg()
            createdb[
                "-U",
                self._db_user,
                "-h",
                self._cluster_host,
                "-p",
                self._cluster_port,
                self._db_name,
            ].run_fg()

        # Run PGTune.
        self._pgtune()

        # Restore state from the spec.
        self._restore_from_state()

    def delete_target_dbms(self):
        return
        sudo[pg_dropcluster["--stop", self._cluster_version, self._cluster_name]].run()

    def _start_target_dbms(self):
        sudo[pg_ctlcluster[self._cluster_version, self._cluster_name, "start"]].run()
        self._wait_until_ready()

    def _stop_target_dbms(self):
        sudo[pg_ctlcluster[self._cluster_version, self._cluster_name, "stop"]].run()
        self._wait_until_ready()

    def _restart_target_dbms(self):
        sudo[pg_ctlcluster[self._cluster_version, self._cluster_name, "restart"]].run()
        self._wait_until_ready()

    def _create_cluster(self):
        pg_createcluster_args = [
            self._cluster_version,
            self._cluster_name,
            "-d",
            self._cluster_data,
            "-l",
            self._cluster_log,
            "-p",
            self._cluster_port,
        ]
        sudo[pg_createcluster[pg_createcluster_args]].run()

    def _test_cluster_exists(self):
        retcode, stdout, _ = pg_lsclusters["-j"].run()
        assert retcode == 0, "Couldn't list existing clusters."
        clusters = json.loads(stdout)
        cluster_exists = any(
            [cluster["cluster"] == "gymcluster" for cluster in clusters]
        )
        return cluster_exists

    def _run_sql(self, sql, as_postgres=False):
        if as_postgres:
            psql_command = psql[
                "-p",
                self._cluster_port,
                "-c",
                sql,
            ]
            sudo["-u", "postgres", "--login", psql_command].run()
        else:
            with local.env(PGPASSWORD=self._db_pass):
                psql_command = psql[
                    "-p",
                    self._cluster_port,
                    "-h",
                    self._cluster_host,
                    "-U",
                    self._db_user,
                    "-d",
                    self._db_name,
                    "-c",
                    sql,
                ]
                psql_command.run()

    def _pgtune(self):
        """
        Set pgtune configuration.

        TODO(WAN): currently hardcoded.

        https://pgtune.leopard.in.ua/
        # DB Version: 14
        # OS Type: linux
        # DB Type: web
        # Total Memory (RAM): 48 GB
        # CPUs num: 6
        # Data Storage: ssd
        """
        sqls = [
            "ALTER SYSTEM SET max_connections = '200'",
            "ALTER SYSTEM SET shared_buffers = '12GB'",
            "ALTER SYSTEM SET effective_cache_size = '36GB'",
            "ALTER SYSTEM SET maintenance_work_mem = '2GB'",
            "ALTER SYSTEM SET checkpoint_completion_target = '0.9'",
            "ALTER SYSTEM SET wal_buffers = '16MB'",
            "ALTER SYSTEM SET default_statistics_target = '100'",
            "ALTER SYSTEM SET random_page_cost = '1.1'",
            "ALTER SYSTEM SET effective_io_concurrency = '200'",
            "ALTER SYSTEM SET work_mem = '20971kB'",
            "ALTER SYSTEM SET min_wal_size = '1GB'",
            "ALTER SYSTEM SET max_wal_size = '4GB'",
            "ALTER SYSTEM SET max_worker_processes = '6'",
            "ALTER SYSTEM SET max_parallel_workers_per_gather = '3'",
            "ALTER SYSTEM SET max_parallel_workers = '6'",
            "ALTER SYSTEM SET max_parallel_maintenance_workers = '3'",
        ]
        for sql in sqls:
            self._run_sql(sql)
        self._restart_target_dbms()

    def _wait_until_ready(self, timeout_s=30, wait_s=5):
        waited_s = 0
        with local.env(PGPASSWORD=self._db_pass):
            while True:
                retcode, _, _ = pg_isready[
                    "-h",
                    self._cluster_host,
                    "-p",
                    self._cluster_port,
                    "-d",
                    self._db_name,
                ].run()
                if retcode == 0:
                    break
                sleep(wait_s)
                waited_s += wait_s
                if waited_s >= timeout_s:
                    raise RuntimeError("pg_isready failed.")

    def _restore_from_state(self):
        assert isinstance(
            self._gym_spec.historical_state, PostgresState
        ), "Invalid state."
        pg_state: PostgresState = self._gym_spec.historical_state

        with local.env(PGPASSWORD=self._db_pass):
            pg_restore[
                "--no-owner",
                "-h",
                self._cluster_host,
                "-p",
                self._cluster_port,
                "-U",
                self._db_user,
                "-d",
                self._db_name,
                "--clean",
                "--if-exists",
                # Leaving this here to note that YOU DO NOT WANT THE CREATE OPTION.
                # Otherwise, it won't restore into the right database.
                # "--create",
                "--exit-on-error",
                "-j",
                psutil.cpu_count(logical=True),
                "--verbose",
                pg_state._historical_state_path,
            ].run()

        # TODO(WAN): This is a hacky way of going about things.
        if self._gym_spec.snapshot is None:
            engine = create_engine(self.get_target_dbms_connstr_sqlalchemy())
            inspector = inspect(engine)
            self._gym_spec.snapshot_db(engine, inspector)
