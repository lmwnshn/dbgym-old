from abc import ABC
import pglast
from pathlib import Path


class Workload(ABC):
    def __init__(self, queries: list[str]):
        self.queries: list[str] = queries


class WorkloadTPCH(Workload):
    def __init__(self, seed):
        queries = []
        seed_path = Path(f"/tpch_queries/{seed}/")
        for i in range(1, 22 + 1):
            with open(seed_path / f"{i}.sql") as f:
                contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
                for sql in pglast.split(contents):
                    queries.append(sql)
        super().__init__(queries)
