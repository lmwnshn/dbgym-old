from flask import Blueprint, abort, current_app, g, request
from plumbum import local
import os
import contextlib
import shutil
import sqlite3
import time
import psutil
import filecmp

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
        command = local["make"]["install-world-bin", "-j"]
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
    stdout_file = bin_dir / "pg_stdout.txt"
    stderr_file = bin_dir / "pg_stderr.txt"

    if not bin_dir.exists():
        abort(404)

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS db ("
            "port INTEGER PRIMARY KEY, "
            "initialized INTEGER NOT NULL, "
            "stdin_file TEXT NOT NULL, "
            "stdout_file TEXT NOT NULL, "
            "stderr_file TEXT NOT NULL,"
            "pid INTEGER)"
        )
        db = conn.execute("SELECT * FROM db WHERE port = ?", (port,)).fetchone()
        if db is not None and db["pid"] is not None:
            return {"message": f"Server on port {port} already running: {tuple(db)}."}

        with open(stdin_file, "w") as stdin, open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
            command = local[str(bin_dir / "postgres")]["-D", bin_dir / "pgdata", "-p", port]
            pg = command.run_bg(stdin=stdin, stdout=stdout, stderr=stderr)
        if db is None:
            conn.execute("INSERT INTO db VALUES (?,?,?,?,?,?)", (port, 0, str(stdin_file), str(stdout_file), str(stderr_file), pg.proc.pid))
        else:
            conn.execute("UPDATE db SET pid = ? WHERE port = ?", (pg.proc.pid, port))
        result = {"pg": {"pid": pg.proc.pid, "stdin": str(stdin_file), "stdout": str(stdout_file), "stderr": str(stderr_file)}}

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

    if not db_file.exists():
        abort(404)

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        db = conn.execute("SELECT * FROM db WHERE port = ?", (port,)).fetchone()
        if db is None or db["pid"] is None:
            return {"message": f"No pid for port {port}: {tuple(db)}"}
        pid = db["pid"]
        command = local["kill"]["-INT", pid]
        result = run_command(command, retcode=[0, 1])
        conn.execute("UPDATE db SET pid = NULL WHERE port = ?", (port,))

        while (bin_dir / "pgdata" / "postmaster.pid").exists():
            time.sleep(1)

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


@postgres.route("/pg_auto_conf/")
def pg_auto_conf():
    conf_path = request.args.get("conf_path")
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    dest_path = bin_dir / "pgdata" / "postgresql.auto.conf"

    if not dest_path.exists():
        abort(404)

    if not filecmp.cmp(conf_path, dest_path):
        shutil.copyfile(conf_path, dest_path)
        return {"status": "overwrote " + str(dest_path)}
    return {"status": "noop"}



@postgres.route("/clean/")
def clean():
    shutil.rmtree(current_app.config["BUILD_DIR"] / "postgres", ignore_errors=True)
    return {"deleted": str(current_app.config["BUILD_DIR"] / "postgres")}


@postgres.route("/nyoom_install/")
def nyoom_install():
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    dest_path = bin_dir / "pgdata" / "postgresql.auto.conf"

    buffer = []
    with open(dest_path, "r") as f:
        for line in f:
            if line.startswith("shared_preload_libraries"):
                key, val, _ = line.split("'")
                if "nyoom" not in val:
                    val = f"{val},nyoom"
                buffer.append(f"{key}'{val}'")
            else:
                buffer.append(line)
        else:
            buffer.append("shared_preload_libraries='nyoom'")
    with open(dest_path, "w") as f:
        for line in buffer:
            print(line, file=f)

    with tmp_cwd(current_app.config["BUILD_DIR"] / "postgres" / "cmudb" / "extensions" / "nyoom"):
        command = local["make"]["clean"]
        _ = run_command(command)
        command = local["make"]["install", "-j"]
        return run_command(command)

@postgres.route("/nyoom_start/")
def nyoom_start():
    db_name = request.args.get("db_name", default=PG_DB)
    db_user = request.args.get("db_user", default=PG_USER)
    db_pass = request.args.get("db_pass", default=PG_PASS)
    port = request.args.get("port", default=PG_PORT)

    nyoom_script = current_app.config["BUILD_DIR"] / "postgres" / "cmudb" / "extensions" / "nyoom" / "nyoom.py"
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    db_file = bin_dir / "trainer.db"
    stdin_file = bin_dir / "nyoom_stdin.txt"
    stdout_file = bin_dir / "nyoom_stdout.txt"
    stderr_file = bin_dir / "nyoom_stderr.txt"

    if not bin_dir.exists():
        abort(404)

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row

        conn.execute(
            "CREATE TABLE IF NOT EXISTS nyoom ("
            "port INTEGER PRIMARY KEY, "
            "stdin_file TEXT NOT NULL, "
            "stdout_file TEXT NOT NULL, "
            "stderr_file TEXT NOT NULL,"
            "pid INTEGER)"
        )
        nyoom_sql = conn.execute("SELECT * FROM nyoom WHERE port = ?", (port,)).fetchone()
        if nyoom_sql is not None and nyoom_sql["pid"] is not None:
            return {"message": f"Nyoom already active on {port}: {tuple(nyoom_sql)}."}

        with open(stdin_file, "w") as stdin, open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
            command = local["/home/wanshenl/.venvs/default/bin/python3"][nyoom_script, "--autoroot", 0]
            nyoom_py = command.run_bg(stdin=stdin, stdout=stdout, stderr=stderr)
        if nyoom_sql is None:
            conn.execute("INSERT INTO nyoom VALUES (?,?,?,?,?)", (port, str(stdin_file), str(stdout_file), str(stderr_file), nyoom_py.proc.pid))
        else:
            conn.execute("UPDATE nyoom SET pid = ? WHERE port = ?", (nyoom_py.proc.pid, port))
        result = {"nyoom": {"pid": nyoom_py.proc.pid, "stdin": str(stdin_file), "stdout": str(stdout_file), "stderr": str(stderr_file)}}

        return result


@postgres.route("/nyoom_stop/")
def nyoom_stop():
    port = request.args.get("port", default=PG_PORT)
    bin_dir = current_app.config["BUILD_DIR"] / "postgres" / "build" / "postgres" / "bin"
    db_file = bin_dir / "trainer.db"

    if not db_file.exists():
        abort(404)

    with sqlite3.connect(db_file) as conn:
        conn.row_factory = sqlite3.Row
        nyoom_sql = conn.execute("SELECT * FROM nyoom WHERE port = ?", (port,)).fetchone()
        if nyoom_sql is None or nyoom_sql["pid"] is None:
            return {"message": f"No pid for port {port}: {tuple(nyoom_sql)}"}
        pid = nyoom_sql["pid"]
        command = local["kill"][pid]
        result = run_command(command, retcode=[0, 1])
        conn.execute("UPDATE nyoom SET pid = NULL WHERE port = ?", (port,))
        return result
