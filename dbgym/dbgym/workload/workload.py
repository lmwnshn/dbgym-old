from __future__ import annotations

import copy
from abc import ABC
from pathlib import Path

import pglast


class Workload(ABC):
    def __init__(self, queries: list[str]):
        self.queries: list[str] = queries


class WorkloadDSB(Workload):
    def __init__(self, config, seed):
        self._config = config
        self._seed = seed
        queries = []
        seed_path = Path(f"/dsb_queries/{config}/{seed}/")
        for sql_path in sorted(seed_path.glob("*.sql"), key=lambda s: str(s).split("-")):
            with open(sql_path) as f:
                contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
                for sql in pglast.split(contents):
                    queries.append(sql)
        super().__init__(queries)

    def __str__(self):
        return f"[DSB-{self._seed}]"

    def __repr__(self):
        return self.__str__()


class WorkloadTPCH(Workload):
    def __init__(self, seed):
        self._seed = seed
        queries = []
        seed_path = Path(f"/tpch_queries/{seed}/")
        for i in range(1, 22 + 1):
            with open(seed_path / f"{i}.sql") as f:
                contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
                for sql in pglast.split(contents):
                    queries.append(sql)
        super().__init__(queries)

    def __str__(self):
        return f"[TPCH-{self._seed}]"

    def __repr__(self):
        return self.__str__()


class WorkloadTPCH(Workload):
    def __init__(self, seed):
        self._seed = seed
        queries = []
        seed_path = Path(f"/tpch_queries/{seed}/")
        for i in range(1, 22 + 1):
            with open(seed_path / f"{i}.sql") as f:
                contents = "".join([line for line in f if not line.startswith("--") and not len(line.strip()) == 0])
                for sql in pglast.split(contents):
                    queries.append(sql)
        super().__init__(queries)

    def __str__(self):
        return f"[TPCH-{self._seed}]"

    def __repr__(self):
        return self.__str__()


class WorkloadNaiveTablesampleTPCH(Workload):
    def __init__(self, workload: WorkloadTPCH):
        queries = copy.deepcopy(workload.queries)
        sample_method = "BERNOULLI (10)"
        sample_seed = "REPEATABLE (15721)"
        for i, query in enumerate(queries, 1):
            if i == 1:
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 2:
                query = query.replace("\n\tpart,", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tpartsupp,", f"\n\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\tnation,", f"\n\t\t\tnation TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\t\t\tregion", f"\n\t\t\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 3:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 4:
                query = query.replace("\n\torders", f"\n\torders TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\tlineitem", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 5:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 6:
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 7:
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n2", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed}"
                )
            elif i == 8:
                query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n2,", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\tregion", f"\n\t\t\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 9:
                query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tpartsupp,", f"\n\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\t\t\tnation", f"\n\t\t\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 10:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 11:
                query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace(
                    "\n\t\t\t\tpartsupp,", f"\n\t\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\t\tsupplier,", f"\n\t\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\t\tnation", f"\n\t\t\t\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 12:
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 13:
                query = query.replace(
                    "\n\t\t\tcustomer left outer join orders",
                    f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed} left outer join orders TABLESAMPLE {sample_method} {sample_seed}",
                )
            elif i == 14:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 15:
                query = query.replace("\n\t\tlineitem", f"\n\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 15 + 1:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            elif i == 15 + 2:
                pass
            elif i == 16 + 2:
                query = query.replace("\n\tpartsupp,", f"\n\tPARTSUPP TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("PARTSUPP", "partsupp")
                query = query.replace("\n\t\t\tsupplier", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 17 + 2:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\tlineitem", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 18 + 2:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\tlineitem", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 19 + 2:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 20 + 2:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\tpartsupp", f"\n\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("\n\t\t\t\t\tpart", f"\n\t\t\t\t\tpart TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace(
                    "\n\t\t\t\t\tlineitem", f"\n\t\t\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}"
                )
            elif i == 21 + 2:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem l1,", f"\n\tlineitem l1 TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace(
                    "\n\t\t\tlineitem l2", f"\n\t\t\tlineitem l2 TABLESAMPLE {sample_method} {sample_seed}"
                )
                query = query.replace(
                    "\n\t\t\tlineitem l3", f"\n\t\t\tlineitem l3 TABLESAMPLE {sample_method} {sample_seed}"
                )
            elif i == 22 + 2:
                query = query.replace("\n\t\t\tcustomer", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace(
                    "\n\t\t\t\t\tcustomer", f"\n\t\t\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed}"
                )
                query = query.replace(
                    "\n\t\t\t\t\torders", f"\n\t\t\t\t\torders TABLESAMPLE {sample_method} {sample_seed}"
                )
            queries[i - 1] = query
        self._seed = workload._seed
        super().__init__(queries)

    def __str__(self):
        return f"[TPCH-NTS-{self._seed}]"

    def __repr__(self):
        return self.__str__()


class WorkloadSmartTablesampleTPCH(Workload):
    def __init__(self, workload: WorkloadTPCH):
        queries = copy.deepcopy(workload.queries)
        # TABLESAMPLE only at root-level to try to limit the perf impact caused by PostgreSQL having a poor optimizer.
        sample_method = "BERNOULLI (10)"
        sample_seed = "REPEATABLE (15721)"
        for i, query in enumerate(queries, 1):
            if i == 1:
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 2:
                query = query.replace("\n\tpart,", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 3:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 4:
                query = query.replace("\n\torders", f"\n\torders TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 5:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation,", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tregion", f"\n\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 6:
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 7:
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n2", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed}"
                )
            elif i == 8:
                query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tcustomer,", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n1,", f"\n\t\t\tnation n1 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tnation n2,", f"\n\t\t\tnation n2 TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\tregion", f"\n\t\t\tregion TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 9:
                query = query.replace("\n\t\t\tpart,", f"\n\t\t\tpart TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace(
                    "\n\t\t\tsupplier,", f"\n\t\t\tsupplier TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tlineitem,", f"\n\t\t\tlineitem TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace(
                    "\n\t\t\tpartsupp,", f"\n\t\t\tpartsupp TABLESAMPLE {sample_method} {sample_seed},"
                )
                query = query.replace("\n\t\t\torders,", f"\n\t\t\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\t\t\tnation", f"\n\t\t\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 10:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 11:
                query = query.replace("\n\tpartsupp,", f"\n\tpartsupp TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 12:
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 13:
                query = query.replace(
                    "\n\t\t\tcustomer left outer join orders",
                    f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed} left outer join orders TABLESAMPLE {sample_method} {sample_seed}",
                )
            elif i == 14:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 15:
                query = query.replace("\n\t\tlineitem", f"\n\t\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 15 + 1:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
            elif i == 15 + 2:
                pass
            elif i == 16 + 2:
                query = query.replace("\n\tpartsupp,", f"\n\tPARTSUPP TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
                query = query.replace("PARTSUPP", "partsupp")
            elif i == 17 + 2:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 18 + 2:
                query = query.replace("\n\tcustomer,", f"\n\tcustomer TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 19 + 2:
                query = query.replace("\n\tlineitem,", f"\n\tlineitem TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tpart", f"\n\tpart TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 20 + 2:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 21 + 2:
                query = query.replace("\n\tsupplier,", f"\n\tsupplier TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tlineitem l1,", f"\n\tlineitem l1 TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\torders,", f"\n\torders TABLESAMPLE {sample_method} {sample_seed},")
                query = query.replace("\n\tnation", f"\n\tnation TABLESAMPLE {sample_method} {sample_seed}")
            elif i == 22 + 2:
                query = query.replace("\n\t\t\tcustomer", f"\n\t\t\tcustomer TABLESAMPLE {sample_method} {sample_seed}")
            queries[i - 1] = query
        self._seed = workload._seed
        super().__init__(queries)

    def __str__(self):
        return f"[TPCH-STS-{self._seed}]"

    def __repr__(self):
        return self.__str__()
