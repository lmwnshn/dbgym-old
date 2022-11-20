from pathlib import Path


class State:
    def __init__(self, historical_state_path: Path):
        self.historical_state_path: Path = historical_state_path

class PostgresState(State):
    def __init__(self, historical_state_path: Path):
        super().__init__(historical_state_path)
