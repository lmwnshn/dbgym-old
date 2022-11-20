from abc import ABC
from dbgym.envs.gym_spec import GymSpec

import requests


class Trainer(ABC):
    def __init__(self, service_url: str, gym_spec: GymSpec, seed: int = 15721):
        self._service_url = service_url
        self._gym_spec: GymSpec = gym_spec
        self._seed: int = seed
        self.dirty = False

    def dbms_bootstrap(self):
        raise NotImplementedError

    def dbms_init(self):
        raise NotImplementedError

    def dbms_start(self):
        raise NotImplementedError

    def dbms_stop(self):
        raise NotImplementedError

    def dbms_restart(self):
        raise NotImplementedError

    def dbms_connstr(self) -> str:
        raise NotImplementedError

    def dbms_restore(self):
        raise NotImplementedError

    @staticmethod
    def _remove_unset_params(params) -> dict:
        return {k: v for k, v in params.items() if v is not None}

    def run_targets(self, targets):
        responses = []
        for url, params in targets:
            params = self._remove_unset_params(params)
            r = requests.get(url, params=params)
            if r.status_code == 404:
                raise FileNotFoundError("Binary not found?")
            responses.append(r.json())
        return responses


class PostgresTrainer(Trainer):
    def __init__(
            self, service_url: str, gym_spec: GymSpec, seed: int = 15721,
            gh_user=None, gh_repo=None, branch=None, build_type=None,
            db_name=None, db_user=None, db_pass=None, host=None, port=None,
    ):
        super().__init__(service_url, gym_spec, seed)
        self._gh_user = gh_user
        self._gh_repo = gh_repo
        self._branch = branch
        self._build_type = build_type
        self._db_name = db_name
        self._db_user = db_user
        self._db_pass = db_pass
        self._host = host
        self._port = port

    def __enter__(self):
        try:
            self.dbms_start()
        except FileNotFoundError:
            self.dbms_bootstrap()
            self.dbms_init()
            self.dbms_start()
            self.dbms_restore()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dbms_stop()

    def dbms_bootstrap(self):
        targets = [
            (self._service_url + "/postgres/clean/", {}),
            (self._service_url + "/postgres/clone/", {
                "gh_user": self._gh_user,
                "gh_repo": self._gh_repo,
                "branch": self._branch,
            }),
            (self._service_url + "/postgres/configure/", {
                "build_type": self._build_type,
            }),
            (self._service_url + "/postgres/make/", {}),
        ]
        return self.run_targets(targets)

    def dbms_init(self):
        targets = [
            (self._service_url + "/postgres/initdb/", {}),
        ]
        return self.run_targets(targets)

    def dbms_start(self):
        targets = [
            (self._service_url + "/postgres/start/", {
                "db_name": self._db_name,
                "db_user": self._db_user,
                "db_pass": self._db_pass,
                "port": self._port,
            }),
        ]
        return self.run_targets(targets)

    def dbms_stop(self):
        targets = [
            (self._service_url + "/postgres/stop/", {
                "port": self._port,
            }),
        ]
        return self.run_targets(targets)

    def dbms_restart(self):
        self.dbms_stop()
        self.dbms_start()

    def dbms_connstr(self) -> str:
        targets = [
            (self._service_url + "/postgres/connstring/", {
                "db_name": self._db_name,
                "db_user": self._db_user,
                "db_pass": self._db_pass,
                "host": self._host,
                "port": self._port,
            }),
        ]
        return self.run_targets(targets)[0]["sqlalchemy"]

    def dbms_restore(self):
        targets = [
            (self._service_url + "/postgres/pg_restore/", {
                "db_name": self._db_name,
                "db_user": self._db_user,
                "db_pass": self._db_pass,
                "host": self._host,
                "port": self._port,
                "state_path": str(self._gym_spec.historical_state.historical_state_path),
            }),
        ]
        return self.run_targets(targets)
