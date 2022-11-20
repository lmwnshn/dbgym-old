from dbgym.envs.state import State
from dbgym.envs.workload import Workload


class GymSpec:
    def __init__(
        self,
        historical_workloads: list[Workload],
        historical_state: State,
    ):
        self.historical_workloads: list[Workload] = historical_workloads
        self.historical_state: State = historical_state
