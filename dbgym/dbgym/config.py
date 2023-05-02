import os
from pathlib import Path

gym_db_name = os.getenv("GYM_DB_NAME")
gym_db_pass = os.getenv("GYM_DB_PASS")
gym_db_user = os.getenv("GYM_DB_USER")

trainer_port = os.getenv("TRAINER_PORT")
trainer_pg_name = os.getenv("TRAINER_PG_NAME")
trainer_pg_pass = os.getenv("TRAINER_PG_PASS")
trainer_pg_port = os.getenv("TRAINER_PG_PORT")
trainer_pg_user = os.getenv("TRAINER_PG_USER")

nyoom_port = os.getenv("NYOOM_PORT")

hostname = os.getenv("HOSTNAME")


class Config:
    SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{gym_db_user}:{gym_db_pass}@gym_db/{gym_db_name}"
    TRAINER_URL = f"http://trainer:{trainer_port}"
    TRAINER_PG_URI = (
        f"postgresql+psycopg://{trainer_pg_user}:{trainer_pg_pass}@trainer:{trainer_pg_port}/{trainer_pg_name}"
    )
    NYOOM_URL = f"http://nyoom:{nyoom_port}"

    SAVE_PATH_BASE = Path("/dbgym/artifact/").absolute()
    SAVE_PATH_OBSERVATION = (SAVE_PATH_BASE / "observation").absolute()
    SAVE_PATH_MODEL = (SAVE_PATH_BASE / "model").absolute()
    SAVE_PATH_EVAL = (SAVE_PATH_BASE / "eval").absolute()
    SAVE_PATH_PLOT = (SAVE_PATH_BASE / "plot").absolute()

    HOSTNAME = hostname
    # The below settings are determined by hostname.
    PGTUNE_STATEMENTS = None
    TPCH_DATA = None
    WORKLOAD_SEED_START = None
    WORKLOAD_SEED_END = None
    AUTOGLUON_TIME_LIMIT_S = None


if Config.HOSTNAME == "kapipad":
    Config.PGTUNE_STATEMENTS = [
        # DB Version: 15
        # OS Type: linux
        # DB Type: dw
        # Total Memory (RAM): 16 GB
        # CPUs num: 8
        # Connections num: 20
        # Data Storage: ssd
        "ALTER SYSTEM SET max_connections = '20';",
        "ALTER SYSTEM SET shared_buffers = '4GB';",
        "ALTER SYSTEM SET effective_cache_size = '12GB';",
        "ALTER SYSTEM SET maintenance_work_mem = '2GB';",
        "ALTER SYSTEM SET checkpoint_completion_target = '0.9';",
        "ALTER SYSTEM SET wal_buffers = '16MB';",
        "ALTER SYSTEM SET default_statistics_target = '500';",
        "ALTER SYSTEM SET random_page_cost = '1.1';",
        "ALTER SYSTEM SET effective_io_concurrency = '200';",
        "ALTER SYSTEM SET work_mem = '26214kB';",
        "ALTER SYSTEM SET min_wal_size = '4GB';",
        "ALTER SYSTEM SET max_wal_size = '16GB';",
        "ALTER SYSTEM SET max_worker_processes = '8';",
        "ALTER SYSTEM SET max_parallel_workers_per_gather = '4';",
        "ALTER SYSTEM SET max_parallel_workers = '8';",
        "ALTER SYSTEM SET max_parallel_maintenance_workers = '4';",
    ]
    Config.TPCH_DATA = Path("/tpch_sf1").absolute()
    Config.WORKLOAD_SEED_START = 15721
    Config.WORKLOAD_SEED_END = 15820  # 15730
    Config.AUTOGLUON_TIME_LIMIT_S = 300  # 10
elif Config.HOSTNAME in ["dev8", "dev9"]:
    Config.PGTUNE_STATEMENTS = [
        # WARNING
        # this tool not being optimal
        # for very high memory systems
        # DB Version: 15
        # OS Type: linux
        # DB Type: dw
        # Total Memory (RAM): 160 GB
        # CPUs num: 80
        # Connections num: 20
        # Data Storage: ssd
        "ALTER SYSTEM SET max_connections = '20';",
        "ALTER SYSTEM SET shared_buffers = '40GB';",
        "ALTER SYSTEM SET effective_cache_size = '120GB';",
        "ALTER SYSTEM SET maintenance_work_mem = '2GB';",
        "ALTER SYSTEM SET checkpoint_completion_target = '0.9';",
        "ALTER SYSTEM SET wal_buffers = '16MB';",
        "ALTER SYSTEM SET default_statistics_target = '500';",
        "ALTER SYSTEM SET random_page_cost = '1.1';",
        "ALTER SYSTEM SET effective_io_concurrency = '200';",
        "ALTER SYSTEM SET work_mem = '26214kB';",
        "ALTER SYSTEM SET min_wal_size = '4GB';",
        "ALTER SYSTEM SET max_wal_size = '16GB';",
        "ALTER SYSTEM SET max_worker_processes = '80';",
        "ALTER SYSTEM SET max_parallel_workers_per_gather = '40';",
        "ALTER SYSTEM SET max_parallel_workers = '80';",
        "ALTER SYSTEM SET max_parallel_maintenance_workers = '4';",
    ]
    Config.TPCH_DATA = Path("/tpch_sf100").absolute()
    Config.WORKLOAD_SEED_START = 15721
    Config.WORKLOAD_SEED_END = 15820  # 16720
    Config.AUTOGLUON_TIME_LIMIT_S = 60 * 5
else:
    raise RuntimeError("Customize for your host.")
