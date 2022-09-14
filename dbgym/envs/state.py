class State:
    pass

class PostgresState:
    def __init__(self, historical_state_path):
        self._historical_state_path = historical_state_path