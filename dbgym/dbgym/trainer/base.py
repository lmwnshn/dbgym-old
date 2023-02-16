from abc import ABC

import requests


class BaseTrainer(ABC):
    def __init__(self, service_url: str):
        self._service_url = service_url
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
