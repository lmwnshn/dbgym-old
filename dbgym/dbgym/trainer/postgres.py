import requests
from dbgym.trainer.base import BaseTrainer
from dbgym.db_config import PgConfig


class PostgresTrainer(BaseTrainer):
    def __init__(self, service_url: str, pg_config: PgConfig, force_rebuild=False):
        self.pg_config = pg_config
        self.dirty = False
        super().__init__(service_url=service_url, force_rebuild=force_rebuild)

    def __enter__(self):
        if self.dbms_pull_maybe_remake():
            if self.dbms_db_exists():
                self.dbms_stop()
                self.dbms_start()

        if self.force_rebuild or not self.dbms_bin_exists():
            self.dbms_bootstrap()
        if not self.dbms_db_exists():
            self.dbms_init()
        isready_retcode = self.dbms_isready()
        if isready_retcode == 0:
            # Ready.
            pass
        elif isready_retcode == 1:
            # Starting up, wait.
            self.dbms_isready_blocking()
        elif isready_retcode == 2:
            # Not responding or not started, force stop and start again.
            self.dbms_stop()
            self.dbms_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dbms_stop()

    def dbms_bin_exists(self):
        print("dbms_bin_exists")
        return requests.post(self._service_url + "/postgres/bin_exists/").json()["initialized"]

    def dbms_db_exists(self):
        print("dbms_db_exists")
        json_args = {"db_name": self.pg_config.db_name}
        return requests.post(self._service_url + "/postgres/db_exists/", json=json_args).json()["initialized"]

    def dbms_isready(self):
        print("dbms_isready")
        json_args = {"db_port": self.pg_config.db_port}
        return requests.post(self._service_url + "/postgres/pg_isready/", json=json_args).json()["retcode"]

    def dbms_isready_blocking(self):
        print("dbms_isready_blocking")
        json_args = {"db_port": self.pg_config.db_port}
        requests.post(self._service_url + "/postgres/pg_isready_blocking/", json=json_args)

    def dbms_bootstrap(self):
        print("dbms_bootstrap")
        requests.post(self._service_url + "/postgres/clone/")
        requests.post(self._service_url + "/postgres/configure/")
        requests.post(self._service_url + "/postgres/make/")

    def dbms_init(self):
        print("dbms_init")
        json_args = {"db_name": self.pg_config.db_name}
        requests.post(self._service_url + "/postgres/initdb/", json=json_args)

    def dbms_start(self):
        print("dbms_start")
        json_args = {
            "db_name": self.pg_config.db_name,
            "db_port": self.pg_config.db_port,
            "db_pass": self.pg_config.db_pass,
            "db_user": self.pg_config.db_user,
        }
        requests.post(self._service_url + "/postgres/start/", json=json_args)

    def dbms_stop(self):
        print("dbms_stop")
        json_args = {
            "db_port": self.pg_config.db_port,
        }
        requests.post(self._service_url + "/postgres/stop/", json=json_args)

    def dbms_restart(self):
        print("dbms_restart")
        self.dbms_stop()
        self.dbms_start()

    def dbms_pull_maybe_remake(self):
        print("dbms_pull_maybe_remake")
        result = requests.post(self._service_url + "/postgres/pull_maybe_remake/")
        result_json = result.json()
        if "git_hash" in result_json:
            print("dbms_git_hash", result_json["git_hash"].strip())
        return result_json["remake"]

    def dbms_install_nyoom(self):
        print("dbms_install_nyoom")
        json_args = {"db_port": self.pg_config.db_port}
        requests.post(self._service_url + "/postgres/nyoom/", json=json_args)
        self.dbms_restart()
