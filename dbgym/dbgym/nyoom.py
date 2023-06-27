import time

import requests
from dbgym.config import Config


def nyoom_start(nyoom_args: dict = None):
    if nyoom_args is None:
        nyoom_args = {}
    req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
    # TODO(WAN): pixie dust
    time.sleep(10)
    req = requests.post(Config.NYOOM_URL + "/nyoom/start/", json=nyoom_args)
    assert req.status_code == 200
    print("nyoom_start: ", req.text)


def nyoom_stop():
    req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
    assert req.status_code == 200
    print("nyoom_stop: ", req.text)
