import os
from pathlib import Path

gym_db_name = os.getenv("GYM_DB_NAME")
gym_db_pass = os.getenv("GYM_DB_PASS")
gym_db_user = os.getenv("GYM_DB_USER")


class Config:
    SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{gym_db_user}:{gym_db_pass}@gym_db/{gym_db_name}"
    TRAINER_DIR = Path("/trainer").absolute()
    TRAINER_DB_DIR = Path("/trainer_db").absolute()
