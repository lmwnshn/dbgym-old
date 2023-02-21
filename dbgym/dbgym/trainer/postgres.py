import requests
from dbgym.trainer.base import BaseTrainer


class PostgresTrainer(BaseTrainer):
    def __enter__(self):
        if not self.dbms_exists():
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
        return requests.post(self._service_url + "/postgres/exists/").json()["initialized"]

    def dbms_isready(self):
        return requests.post(self._service_url + "/postgres/pg_isready/").json()["retcode"]

    def dbms_isready_blocking(self):
        requests.post(self._service_url + "/postgres/pg_isready_blocking/")

    def dbms_bootstrap(self):
        requests.post(self._service_url + "/postgres/clone/")
        requests.post(self._service_url + "/postgres/configure/")
        requests.post(self._service_url + "/postgres/make/")

    def dbms_init(self):
        requests.post(self._service_url + "/postgres/initdb/")

    def dbms_start(self):
        requests.post(self._service_url + "/postgres/start/")

    def dbms_stop(self):
        requests.post(self._service_url + "/postgres/stop/")

    def dbms_restart(self):
        self.dbms_stop()
        self.dbms_start()
