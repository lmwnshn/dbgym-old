from abc import ABC


class DbConfig(ABC):
    def __init__(
            self,
            dataset: str,
            db_name: str,
            hostname: str,
            db_user: str,
            db_pass: str,
            db_port: int,
    ):
        self.hostname = hostname
        self.db_user = db_user
        self.db_pass = db_pass
        self.db_port = db_port
        self.dataset = dataset
        self.db_name = db_name

    def get_uri(self) -> str:
        pass


class PgConfig(DbConfig):
    def get_uri(self):
        return f"postgresql+psycopg://{self.db_user}:{self.db_pass}@{self.hostname}:{self.db_port}/{self.db_name}"
