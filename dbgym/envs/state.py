from pathlib import Path


class State:
    pass


class PostgresState(State):
    def __init__(self, historical_state_path: Path):
        self._historical_state_path: Path = historical_state_path
