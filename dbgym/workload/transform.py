import shutil
from abc import ABC
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect


class WorkloadTransform(ABC):
    def __init__(self, name: str):
        self.name = name

    def transform(self, workload_train_path: Path, workload_test_path: Path, output_db_path: Path):
        raise NotImplementedError

    def __str__(self):
        return self.name


class PerfectPrediction(WorkloadTransform):
    def __init__(self):
        super().__init__(f"PerfectPrediction")

    def transform(self, workload_train_path: Path, workload_test_path: Path, output_db_path: Path):
        shutil.copy(workload_test_path, output_db_path)


class OnlyUniqueQueries(WorkloadTransform):
    def __init__(self):
        super().__init__(f"OnlyUnique")

    def transform(self, workload_db_path: Path, output_db_path: Path):
        shutil.copy(workload_db_path, output_db_path)
        engine = create_engine(f"sqlite:///{output_db_path}")
        df = pd.read_sql("SELECT * FROM workload", engine)
        df = df.iloc[df[["template_id", "params_id"]].drop_duplicates().index]
        df.to_sql("workload", engine, index=False, if_exists="replace")


class SampleFracQueries(WorkloadTransform):
    def __init__(self, frac=0.8, random_state=15721):
        super().__init__(f"SampleFracQueries_frac_{frac}_randomstate_{random_state}")
        self._frac = frac
        self._random_state = random_state

    def transform(self, workload_train_path: Path, workload_test_path: Path, output_db_path: Path):
        shutil.copy(workload_test_path, output_db_path)
        engine = create_engine(f"sqlite:///{output_db_path}")
        df = pd.read_sql("SELECT * FROM workload", engine)
        df = df.sample(frac=self._frac, random_state=self._random_state).sort_index()
        df.to_sql("workload", engine, index=False, if_exists="replace")


class ParamSubstitution(WorkloadTransform):
    def __init__(self, name: str, func, unquote=True, numeric=True):
        assert (
            func is not None
        ), "Provide a substitution function of type pd.Series -> assignable to pd.Series."
        super().__init__(name)
        self._func = func
        self._unquote = unquote
        self._numeric = numeric

    def transform(self, workload_train_path: Path, workload_test_path: Path, output_db_path: Path):
        shutil.copy(workload_test_path, output_db_path)
        engine = create_engine(f"sqlite:///{output_db_path}")
        inspector = inspect(engine)
        for table_name in inspector.get_table_names():
            if table_name.endswith("_params"):
                df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
                for column in df.columns:
                    if column.startswith("param_"):
                        target = df[column]
                        # Note that each parameter is quoted by default.
                        if self._unquote:
                            target = target.str.slice(start=1, stop=-1)
                        if self._numeric:
                            if not (target.str.len() > 2).any():
                                continue
                            try:
                                target = pd.to_numeric(target)
                            except ValueError:
                                continue
                        df[column] = self._func(target)
                df.to_sql(table_name, engine, index=False, if_exists="replace")
