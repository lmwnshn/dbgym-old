import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

from dbgym.envs.workload import Workload
from dbgym.util.postgres_workload import convert_postgresql_csvlog_to_workload
from dbgym.workload.transform import WorkloadTransform


@dataclass
class ExperimentWorkloads:
    name: str
    workloads: list[Workload]


class WorkloadManager:
    def __init__(
        self,
        save_name: str,
        save_path: Path,
        train_csvlog_path: Path,
        test_csvlog_path: Optional[Path] = None,
    ):
        self._transforms = []
        self._save_name = save_name
        self._save_path = save_path
        self._train_csvlog_path = train_csvlog_path
        self._test_csvlog_path = test_csvlog_path

        self._train_db_path = self._save_path / f"{save_name}_train.db"
        self._test_db_path = self._save_path / f"{save_name}_test.db"

    def add_transform(self, transform: WorkloadTransform) -> None:
        self._transforms.append(transform)

    def generate_experiment_workloads(
        self, include_only_train: bool = True, include_only_test: bool = True
    ) -> list[ExperimentWorkloads]:
        experiments: list[ExperimentWorkloads] = []

        # Create .db files for both train and test, if necessary.
        with tqdm(total=2, desc="Processing input CSVLOGs into DB files.", leave=False) as pbar:
            if not self._train_db_path.exists():
                self._train_db_path.parent.mkdir(parents=True, exist_ok=True)
                convert_postgresql_csvlog_to_workload(self._train_csvlog_path, self._train_db_path)
            pbar.update(1)
            if not self._test_db_path.exists():
                self._test_db_path.parent.mkdir(parents=True, exist_ok=True)
                if self._test_csvlog_path is not None:
                    convert_postgresql_csvlog_to_workload(self._test_csvlog_path, self._test_db_path)
                else:
                    self._split_train_db_into_train_and_test_db()
            pbar.update(1)

        num_transforms = int(include_only_test) + int(include_only_train) + len(self._transforms)
        with tqdm(total=num_transforms, desc="Creating transformed DB files.", leave=False) as pbar:
            # TODO(WAN): Hack; for evaluation reasons, OnlyTest should appear first.
            if include_only_test:
                pbar.set_description(f"Transform: OnlyTest")
                experiments.append(ExperimentWorkloads("OnlyTest", [Workload(self._test_db_path)]))
                pbar.update(1)
            if include_only_train:
                pbar.set_description(f"Transform: OnlyTrain")
                experiments.append(ExperimentWorkloads("OnlyTrain", [Workload(self._train_db_path)]))
                pbar.update(1)

            # Create any necessary transforms.
            self._save_path.mkdir(parents=True, exist_ok=True)
            for transform in self._transforms:
                pbar.set_description(f"Transform: {transform.name}")
                transform_path = self._save_path / f"{self._save_name}_{transform.name}.db"
                if not transform_path.exists():
                    transform.transform(self._train_db_path, self._test_db_path, transform_path)
                experiments.append(
                    ExperimentWorkloads(
                        transform.name,
                        [Workload(self._train_db_path), Workload(transform_path)],
                    )
                )
                pbar.update(1)

        return experiments

    def _split_train_db_into_train_and_test_db(self):
        # Copy train DB to test DB.
        shutil.copy(self._train_db_path, self._test_db_path)
        # Find the midpoint based on query number.
        engine = create_engine(f"sqlite:///{self._train_db_path}")
        df = pd.read_sql("SELECT * FROM workload", engine)
        query_nums = sorted(df["query_num"].unique())
        midpoint = query_nums[len(query_nums) // 2]
        # Delete rows from each DB file so that first half is train and second half is test.
        engine.execute(f"DELETE FROM workload WHERE query_num > {midpoint}")
        engine = create_engine(f"sqlite:///{self._test_db_path}")
        engine.execute(f"DELETE FROM workload WHERE query_num <= {midpoint}")
