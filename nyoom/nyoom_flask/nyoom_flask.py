from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Optional

from flask import Blueprint, current_app, request
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
    db_name = None
    db_port = None
    method = None  # "tskip"
    tskip_wiggle_std = None  # 2.0
    tskip_wiggle_sampen = None  # 20
    optimizer_cutoff_pct = None  # 10
    optimizer_min_processed = None  # 0

    req_json = request.get_json(silent=True)
    try:
        db_name = req_json.get("db_name")
        db_port = int(req_json.get("db_port"))
        db_pass = req_json.get("db_pass")
        db_user = req_json.get("db_user")
        method = req_json.get("method")
        if method == "tskip":
            tskip_wiggle_std = req_json.get("tskip_wiggle_std")
            tskip_wiggle_sampen = req_json.get("tskip_wiggle_sampen")
        elif method == "optimizer":
            optimizer_cutoff_pct = req_json.get("optimizer_cutoff_pct")
            optimizer_min_processed = req_json.get("optimizer_min_processed")
    except Exception:
        return f"Bad request: {req_json=}", 400

    startup_args = ["-u", "-m", "nyoom",
                    "--db_name", db_name, "--db_port", db_port,
                    "--db_user", db_user, "--db_pass", db_pass,
                    "--method", f"{method}"]
    suffix = f"{method}"
    if method == "tskip":
        startup_args.extend(
            [
                "--tskip_wiggle_std",
                f"{tskip_wiggle_std}",
                "--tskip_wiggle_sampen",
                f"{tskip_wiggle_sampen}",
            ]
        )
        suffix += f"_std_{tskip_wiggle_std}_sampen_{tskip_wiggle_sampen}"
    elif method == "optimizer":
        startup_args.extend(
            [
                "--optimizer_cutoff_pct",
                f"{optimizer_cutoff_pct}",
                "--optimizer_min_processed",
                f"{optimizer_min_processed}",
            ]
        )
        suffix += f"_cutoff_{optimizer_cutoff_pct}_min_{optimizer_min_processed}"

    query = db.select(NyoomInstance).where(NyoomInstance.port == db_port)
    instance: Optional[NyoomInstance] = db.session.execute(query).scalar_one_or_none()

    if instance is not None and instance.pid is not None:
        return {"message": f"Instance already exists: {instance}"}

    stdin_file = get_nyoom_dir() / f"nyoom_stdin_{suffix}.txt"
    stdout_file = get_nyoom_dir() / f"nyoom_stdout_{suffix}.txt"
    stderr_file = get_nyoom_dir() / f"nyoom_stderr_{suffix}.txt"

    with open(stdin_file, "w") as stdin, open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
        with tmp_cwd(get_nyoom_dir()):
            command = local["python3"][startup_args]
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
    req_json = request.get_json(silent=True)
    try:
        db_port = req_json.get("db_port")
    except Exception:
        return f"Bad request: {req_json=}", 400

    query = db.select(NyoomInstance).where(NyoomInstance.port == db_port)
    instance: Optional[NyoomInstance] = db.session.execute(query).scalar_one_or_none()
    if instance is None or instance.pid is None:
        return {"message": f"No PID for {instance}"}
    command = local["kill"]["-INT", instance.pid]
    result = run_command(command, expected_retcodes=[-2, 0, 1])
    instance.pid = None
    db.session.commit()
    return result
