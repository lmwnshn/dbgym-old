from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Optional

from flask import Blueprint, current_app
from nyoom_flask.extensions import db
from nyoom_flask.model.instance import NyoomInstance
from plumbum import local
from plumbum.machines import LocalCommand

nyoom_flask = Blueprint("nyoom", __name__)


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


def get_nyoom_dir() -> Path:
    nyoom_dir: Path = current_app.config["NYOOM_DIR"]
    return nyoom_dir.absolute()


@nyoom_flask.route("/start/", methods=["POST"])
def start():
    db_port = int(os.getenv("TRAINER_PG_PORT"))

    query = db.select(NyoomInstance).where(NyoomInstance.port == db_port)
    instance: Optional[NyoomInstance] = db.session.execute(query).scalar_one_or_none()

    if instance is not None and instance.pid is not None:
        return {"message": f"Instance already exists: {instance}"}

    stdin_file = get_nyoom_dir() / "nyoom_stdin.txt"
    stdout_file = get_nyoom_dir() / "nyoom_stdout.txt"
    stderr_file = get_nyoom_dir() / "nyoom_stderr.txt"

    with open(stdin_file, "w") as stdin, open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
        with tmp_cwd(get_nyoom_dir()):
            command = local["python3"]["-u", "-m", "nyoom"]
            pg = command.run_bg(stdin=stdin, stdout=stdout, stderr=stderr)
            pid = pg.proc.pid

    result = {
        "nyoom": {
            "pid": pid,
        },
        "stdin": str(stdin_file),
        "stdout": str(stdout_file),
        "stderr": str(stderr_file),
    }

    if instance is not None:
        instance.initialized = True
        instance.pid = pid
    else:
        instance = NyoomInstance(
            port=db_port,
            stdin_file=str(stdin_file),
            stdout_file=str(stdout_file),
            stderr_file=str(stderr_file),
            pid=pid,
        )
        db.session.add(instance)

    db.session.commit()
    return result


@nyoom_flask.route("/stop/", methods=["POST"])
def stop():
    db_port = int(os.getenv("TRAINER_PG_PORT"))
    query = db.select(NyoomInstance).where(NyoomInstance.port == db_port)
    instance: Optional[NyoomInstance] = db.session.execute(query).scalar_one_or_none()
    if instance is None or instance.pid is None:
        return {"message": f"No PID for {instance}"}
    command = local["kill"]["-INT", instance.pid]
    result = run_command(command, expected_retcodes=[-2, 0, 1])
    instance.pid = None
    db.session.commit()
    return result
