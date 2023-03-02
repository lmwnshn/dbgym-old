import requests
from dbgym.trainer.base import BaseTrainer


class PostgresTrainer(BaseTrainer):
    def __enter__(self):
        if self.force_rebuild or not self.dbms_exists():
            self.dbms_bootstrap()
            self.dbms_init()
        isready_retcode = self.dbms_isready()
        if isready_retcode == 0:
            # Ready.
            pass
        elif isready_retcode == 1:
            # Starting up, wait.
            self.dbms_isready_blocking()
        elif isready_retcode == 2:
            # Not responding, force stop and start again.
            self.dbms_stop()
            self.dbms_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dbms_stop()

    def dbms_exists(self):
        print("dbms_exists")
        return requests.post(self._service_url + "/postgres/exists/").json()["initialized"]

    def dbms_isready(self):
        print("dbms_isready")
        return requests.post(self._service_url + "/postgres/pg_isready/").json()["retcode"]

    def dbms_isready_blocking(self):
        print("dbms_isready_blocking")
        requests.post(self._service_url + "/postgres/pg_isready_blocking/")

    def dbms_bootstrap(self):
        print("dbms_bootstrap")
        requests.post(self._service_url + "/postgres/clone/")
        requests.post(self._service_url + "/postgres/configure/")
        requests.post(self._service_url + "/postgres/make/")

    def dbms_init(self):
        print("dbms_init")
        requests.post(self._service_url + "/postgres/initdb/")

    def dbms_start(self):
        print("dbms_start")
        requests.post(self._service_url + "/postgres/start/")

    def dbms_stop(self):
        print("dbms_stop")
        requests.post(self._service_url + "/postgres/stop/")

    def dbms_restart(self):
        print("dbms_restart")
        self.dbms_stop()
        self.dbms_start()

    def dbms_install_nyoom(self):
        print("dbms_install_nyoom")
        requests.post(self._service_url + "/postgres/nyoom/")
        self.dbms_restart()
