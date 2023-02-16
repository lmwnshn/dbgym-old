import os

gym_db_name = os.getenv("GYM_DB_NAME")
gym_db_pass = os.getenv("GYM_DB_PASS")
gym_db_user = os.getenv("GYM_DB_USER")

trainer_port = os.getenv("TRAINER_PORT")
trainer_pg_name = os.getenv("TRAINER_PG_NAME")
trainer_pg_pass = os.getenv("TRAINER_PG_PASS")
trainer_pg_port = os.getenv("TRAINER_PG_PORT")
trainer_pg_user = os.getenv("TRAINER_PG_USER")


class Config:
    SQLALCHEMY_DATABASE_URI = f"postgresql+psycopg://{gym_db_user}:{gym_db_pass}@gym_db/{gym_db_name}"
    TRAINER_URL = f"http://trainer:{trainer_port}"
    TRAINER_PG_URI = f"postgresql+psycopg://{trainer_pg_user}:{trainer_pg_pass}@trainer:{trainer_pg_port}/{trainer_pg_name}"
