from __future__ import annotations

import contextlib
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from flask import Blueprint, current_app, request
from plumbum import local
from plumbum.machines import LocalCommand

from trainer.extensions import db
from trainer.model.instance import Instance

postgres = Blueprint("postgres", __name__)


@contextlib.contextmanager
def tmp_cwd(tmp_wd):
    old_dir = os.getcwd()
    try:
        os.chdir(tmp_wd)
        yield
    finally:
        os.chdir(old_dir)


def run_command(command: LocalCommand, expected_retcodes: int | list[int] | None = 0):
    retcode, stdout, stderr = command.run(retcode=expected_retcodes)
    result = {
        "command": str(command),
        "retcode": retcode,
        "stdout": stdout,
        "stderr": stderr,
    }
    return result


def get_trainer_dir() -> Path:
    trainer_dir: Path = current_app.config["TRAINER_DIR"]
    return trainer_dir.absolute()


def get_pg_dir() -> Path:
    return (get_trainer_dir() / "postgres").absolute()


def get_pg_build_dir() -> Path:
    return (get_pg_dir() / "build" / "postgres").absolute()


def get_pg_bin_dir() -> Path:
    return (get_pg_build_dir() / "bin").absolute()


def get_pg_data_dir() -> Path:
    return (get_pg_bin_dir() / "pgdata").absolute()


def get_nyoom_dir() -> Path:
    return (get_pg_dir() / "cmudb" / "extensions" / "nyoom").absolute()


@postgres.route("/exists/", methods=["POST"])
def exists():
    return {"initialized": get_pg_data_dir().exists()}


@postgres.route("/pg_isready/", methods=["POST"])
def pg_isready():
    db_port = int(os.getenv("TRAINER_PG_PORT"))
    command = local[get_pg_bin_dir() / "pg_isready"]["-p", db_port]
    result = run_command(command, expected_retcodes=[0, 1, 2])
    return result


@postgres.route("/pg_isready_blocking/", methods=["POST"])
def pg_isready_blocking():
    db_port = int(os.getenv("TRAINER_PG_PORT"))
    command = local[get_pg_bin_dir() / "pg_isready"]["-p", db_port]
    while True:
        result = run_command(command, expected_retcodes=[0, 1, 2])
        if result["retcode"] == 0:
            break
        else:
            time.sleep(1)
    return result


@postgres.route("/clone/", methods=["POST"])
def clone():
    gh_user = "lmwnshn"
    gh_repo = "postgres"
    gh_branch = "wan"

    req_json = request.get_json(silent=True)
    if req_json is not None:
        gh_user = req_json.get("gh_user", default=gh_user)
        gh_repo = req_json.get("gh_repo", default=gh_repo)
        gh_branch = req_json.get("gh_branch", default=gh_branch)

    args = ["clone", f"https://github.com/{gh_user}/{gh_repo}.git", "--depth", "1"]
    if gh_branch is not None:
        args.extend(["--single-branch", "--branch", gh_branch])
    with tmp_cwd(get_trainer_dir()):
        local["rm"]["-rf", gh_repo].run()
        db_port = int(os.getenv("TRAINER_PG_PORT"))
        query = db.select(Instance).where(Instance.port == db_port)
        instance: Optional[Instance] = db.session.execute(query).scalar_one_or_none()
        if instance is not None:
            db.session.delete(instance)
            db.session.commit()
        command = local["git"][args, "postgres"]
        return run_command(command)


@postgres.route("/configure/", methods=["POST"])
def configure():
    build_type = "release"

    req_json = request.get_json(silent=True)
    if req_json is not None:
        build_type = req_json.get("build_type", default=build_type)

    config_sh = get_pg_dir() / "cmudb" / "build" / "configure.sh"
    with tmp_cwd(get_pg_dir()):
        command = local[config_sh][build_type, get_pg_build_dir()]
        return run_command(command)


@postgres.route("/make/", methods=["POST"])
def make():
    with tmp_cwd(get_pg_dir()):
        command = local["make"]["install-world-bin", "-j"]
        return run_command(command)


@postgres.route("/initdb/", methods=["POST"])
def initdb():
    shutil.rmtree(get_pg_data_dir(), ignore_errors=True)
    command = local[get_pg_bin_dir() / "initdb"]["-D", get_pg_data_dir()]
    result = run_command(command)
    with tmp_cwd(get_pg_data_dir()):
        with open("pg_hba.conf", "a") as f:
            print("host all all 0.0.0.0/0 md5", file=f)
        with open("postgresql.conf", "a") as f:
            print("listen_addresses = '*'", file=f)
    return result


@postgres.route("/start/", methods=["POST"])
def start():
    db_name = os.getenv("TRAINER_PG_NAME")
    db_port = int(os.getenv("TRAINER_PG_PORT"))
    db_pass = os.getenv("TRAINER_PG_PASS")
    db_user = os.getenv("TRAINER_PG_USER")

    query = db.select(Instance).where(Instance.port == db_port)
    instance: Optional[Instance] = db.session.execute(query).scalar_one_or_none()

    if instance is not None and instance.pid is not None:
        return {"message": f"Instance already exists: {instance}"}

    stdin_file = get_pg_bin_dir() / "pg_stdin.txt"
    stdout_file = get_pg_bin_dir() / "pg_stdout.txt"
    stderr_file = get_pg_bin_dir() / "pg_stderr.txt"

    with open(stdin_file, "w") as stdin, open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
        command = local[get_pg_bin_dir() / "postgres"]["-D", get_pg_data_dir(), "-p", db_port]
        pg = command.run_bg(stdin=stdin, stdout=stdout, stderr=stderr)
        pid = pg.proc.pid

    result = {
        "postgres": {
            "pid": pid,
        },
        "stdin": str(stdin_file),
        "stdout": str(stdout_file),
        "stderr": str(stderr_file),
    }

    pg_isready = local[get_pg_bin_dir() / "pg_isready"]["-p", db_port]
    while True:
        pg_isready_result = run_command(pg_isready, expected_retcodes=[0, 1, 2])
        if pg_isready_result["retcode"] == 0:
            result["pg_isready"] = pg_isready_result
            break
        else:
            time.sleep(1)

    if instance is None or not instance.initialized:
        psql_args_list = [
            ["postgres", "-p", db_port, "-c", f"create user {db_user} with login password '{db_pass}'"],
            ["postgres", "-p", db_port, "-c", f"create database {db_name} with owner = '{db_user}'"],
            ["postgres", "-p", db_port, "-c", f"grant pg_monitor to {db_user}"],
            ["postgres", "-p", db_port, "-c", f"alter user {db_user} with superuser"],
        ]
        psql = local[get_pg_bin_dir() / "psql"]
        result["setup"] = []
        for psql_args in psql_args_list:
            command = psql[psql_args]
            result["setup"].append(run_command(command, expected_retcodes=[0, 1]))

    if instance is not None:
        instance.initialized = True
        instance.pid = pid
    else:
        instance = Instance(
            port=db_port,
            initialized=True,
            db_type="postgres",
            stdin_file=str(stdin_file),
            stdout_file=str(stdout_file),
            stderr_file=str(stderr_file),
            pid=pid,
        )
        db.session.add(instance)

    db.session.commit()
    return result


@postgres.route("/stop/", methods=["POST"])
def stop():
    db_port = int(os.getenv("TRAINER_PG_PORT"))
    query = db.select(Instance).where(Instance.port == db_port)
    instance: Optional[Instance] = db.session.execute(query).scalar_one_or_none()
    if instance is None or instance.pid is None:
        return {"message": f"No PID for {instance}"}
    command = local["kill"]["-INT", instance.pid]
    result = run_command(command, expected_retcodes=[0, 1])
    instance.pid = None
    db.session.commit()
    if result["retcode"] == 0:
        while (get_pg_data_dir() / "postmaster.pid").exists():
            time.sleep(1)
    return result


@postgres.route("/nyoom/", methods=["POST"])
def nyoom():
    db_port = int(os.getenv("TRAINER_PG_PORT"))

    result = {}
    with tmp_cwd(get_nyoom_dir()):
        command = local["make"]["install", "-j"]
        result["make"] = run_command(command)

    psql = local[get_pg_bin_dir() / "psql"]
    psql_args = ["postgres", "-p", db_port, "-c", "ALTER SYSTEM SET shared_preload_libraries = 'nyoom'"]
    command = psql[psql_args]
    result["shared_preload_libraries"] = run_command(command)

    # TODO(WAN): should restart, but we already will restart for pgtune so...
    return result

