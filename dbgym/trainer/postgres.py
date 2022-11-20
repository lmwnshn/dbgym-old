from flask import Blueprint, abort, current_app, g, request
from plumbum import local
import os
import contextlib
import shutil
import sqlite3
import time
import psutil

from dbgym.trainer import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS

postgres = Blueprint('postgres', __name__)


@contextlib.contextmanager
def tmp_cwd(tmp_wd):
    # Python 3.11... :(
    old_dir = os.getcwd()
    try:
        os.chdir(tmp_wd)
        yield
    finally:
        os.chdir(old_dir)


def run_command(command, retcode: int | list[int] | None = 0):
    try:
        retcode, stdout, stderr = command.run(retcode=retcode)
        result = {
            "command": str(command),
            "retcode": retcode,
            "stdout": stdout,
            "stderr": stderr,
        }
        return result
    except FileNotFoundError:
        abort(404)


@postgres.route("/clone/")
def clone():
    gh_user = request.args.get("gh_user", default="cmu-db")
    gh_repo = request.args.get("gh_repo", default="postgres")
    branch = request.args.get("branch", default=None)
    args = ["clone", f"git@github.com:{gh_user}/{gh_repo}.git", "--depth", "1"]
    if branch is not None:
        args.extend(["--single-branch", "--branch", branch])
    current_app.config["BUILD_DIR"].mkdir(parents=True, exist_ok=True)
    with tmp_cwd(current_app.config["BUILD_DIR"]):
        local["rm"]["-rf", gh_repo].run()
        command = local["git"][args]
        return run_command(command)


@postgres.route("/configure/")
def configure():
    build_type = request.args.get("build_type", default="release")
    if build_type not in ["debug", "release"]:
        abort(400)
    with tmp_cwd(current_app.config["BUILD_DIR"] / "postgres"):
        build_path = (current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres").absolute()
        args = [build_type, build_path]
        command = local[str(current_app.config["BUILD_DIR"] / "postgres/cmudb/build/configure.sh")][args]
        return run_command(command)


@postgres.route("/make/")
def make():
    with tmp_cwd(current_app.config["BUILD_DIR"] / "postgres"):
        command = local["make"]["install", "-j"]
        return run_command(command)


@postgres.route("/initdb/")
def initdb():
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    shutil.rmtree(bin_dir / "pgdata", ignore_errors=True)
    command = local[str(bin_dir / "initdb")]["-D", bin_dir / "pgdata"]
    return run_command(command)


@postgres.route("/start/")
def start():
    db_name = request.args.get("db_name", default=PG_DB)
    db_user = request.args.get("db_user", default=PG_USER)
    db_pass = request.args.get("db_pass", default=PG_PASS)
    port = request.args.get("port", default=PG_PORT)

    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    db_file = bin_dir / "trainer.db"
    stdin_file = bin_dir / "pg_stdin.txt"
    stderr_file = bin_dir / "pg_stderr.txt"

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS db ("
            "port INTEGER PRIMARY KEY, "
            "initialized INTEGER NOT NULL, "
            "stdin_file TEXT NOT NULL, "
            "stderr_file TEXT NOT NULL,"
            "pid INTEGER)"
        )
        db = conn.execute("SELECT * FROM db WHERE port = ?", (port,)).fetchone()
        if db is not None and db["pid"] is not None:
            return {"message": f"Server on port {port} already running: {tuple(db)}."}

        with open(stdin_file, "w") as stdin, open(stderr_file, "w") as stderr:
            command = local[str(bin_dir / "postgres")]["-D", bin_dir / "pgdata", "-p", port]
            pg = command.run_bg(stdin=stdin, stderr=stderr)
        if db is None:
            conn.execute("INSERT INTO db VALUES (?,?,?,?,?)", (port, 0, str(stdin_file), str(stderr_file), pg.proc.pid))
        else:
            conn.execute("UPDATE db SET pid = ? WHERE port = ?", (pg.proc.pid, port))
        result = {"pg": {"pid": pg.proc.pid, "stdin": str(stdin_file), "stderr": str(stderr_file)}}

        pg_isready = local[str(bin_dir / "pg_isready")]["-p", port]
        while True:
            pg_isready_result = run_command(pg_isready, retcode=[0, 1, 2])
            if pg_isready_result["retcode"] == 0:
                result["pg_isready"] = pg_isready_result
                break
            else:
                time.sleep(1)

        if db is None or not db["initialized"]:
            psql_args_list = [
                ["postgres", "-p", port, "-c", f"create user {db_user} with login password '{db_pass}'"],
                ["postgres", "-p", port, "-c", f"create database {db_name} with owner = '{db_user}'"],
                ["postgres", "-p", port, "-c", f"grant pg_monitor to {db_user}"],
                ["postgres", "-p", port, "-c", f"alter user {db_user} with superuser"],
            ]
            psql = local[str(bin_dir / "psql")]
            result["setup"] = []
            for psql_args in psql_args_list:
                command = psql[psql_args]
                result["setup"].append(run_command(command, retcode=[0, 1]))
            conn.execute("UPDATE db SET initialized = 1 WHERE port = ?", (port,))

        return result


@postgres.route("/stop/")
def stop():
    port = request.args.get("port", default=PG_PORT)
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    db_file = bin_dir / "trainer.db"
    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        db = conn.execute("SELECT * FROM db WHERE port = ?", (port,)).fetchone()
        if db is None or db["pid"] is None:
            return {"message": f"No pid for port {port}: {tuple(db)}"}
        pid = db["pid"]
        command = local["kill"][pid]
        result = run_command(command, retcode=[0, 1])
        conn.execute("UPDATE db SET pid = NULL WHERE port = ?", (port,))
        return result


@postgres.route("/connstring/")
def connstring():
    db_name = request.args.get("db_name", default=PG_DB)
    db_user = request.args.get("db_user", default=PG_USER)
    db_pass = request.args.get("db_pass", default=PG_PASS)
    host = request.args.get("host", default=PG_HOST)
    port = request.args.get("port", default=PG_PORT)
    return {"sqlalchemy": f"postgresql+psycopg2://{db_user}:{db_pass}@{host}:{port}/{db_name}"}


@postgres.route("/pg_restore/")
def pg_restore():
    state_path = request.args.get("state_path")
    db_name = request.args.get("db_name", default=PG_DB)
    db_user = request.args.get("db_user", default=PG_USER)
    db_pass = request.args.get("db_pass", default=PG_PASS)
    host = request.args.get("host", default=PG_HOST)
    port = request.args.get("port", default=PG_PORT)

    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    with local.env(PGPASSWORD=db_pass):
        args = [
            "--no-owner",
            "-h", host,
            "-p", port,
            "-U", db_user,
            "-d", db_name,
            "--clean",
            "--if-exists",
            # DO NOT USE --create, otherwise it restores into the wrong database.
            "--exit-on-error",
            "-j",
            psutil.cpu_count(logical=True),
            state_path,
        ]
        command = local[str(bin_dir / "pg_restore")][args]
        return run_command(command)


@postgres.route("/clean/")
def clean():
    shutil.rmtree(current_app.config["BUILD_DIR"] / "postgres", ignore_errors=True)
    return {"deleted": str(current_app.config["BUILD_DIR"] / "postgres")}
