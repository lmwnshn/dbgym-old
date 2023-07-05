import time

import requests
from dbgym.config import Config
from dbgym.db_config import DbConfig


def nyoom_start(db_config: DbConfig, nyoom_args: dict):
    req = requests.post(Config.NYOOM_URL + "/nyoom/stop/")
    # TODO(WAN): pixie dust
    time.sleep(10)
    json_args = nyoom_args
    json_args["hostname"] = db_config.hostname
    json_args["db_name"] = db_config.db_name
    json_args["db_port"] = db_config.db_port
    json_args["db_user"] = db_config.db_user
    json_args["db_pass"] = db_config.db_pass
    req = requests.post(Config.NYOOM_URL + "/nyoom/start/", json=json_args)
    assert req.status_code == 200
    print("nyoom_start: ", req.text)


def nyoom_stop(db_config: DbConfig):
    json_args = {"db_port": db_config.db_port}
    req = requests.post(Config.NYOOM_URL + "/nyoom/stop/", json=json_args)
    assert req.status_code == 200
    print("nyoom_stop: ", req.text)
