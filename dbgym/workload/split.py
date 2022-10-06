import shutil
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def split_workload(workload_db_path: Path, train_db_path: Path, test_db_path: Path):
    shutil.copy(workload_db_path, train_db_path)
    shutil.copy(workload_db_path, test_db_path)

    engine = create_engine(f"sqlite:///{train_db_path}")
    df = pd.read_sql("SELECT * FROM workload", engine)
    query_nums = sorted(df["query_num"].unique())
    midpoint = query_nums[len(query_nums) // 2]
    print(f"Midpoint: {midpoint}")
    engine.execute(f"DELETE FROM workload WHERE query_num > {midpoint}")

    engine = create_engine(f"sqlite:///{test_db_path}")
    engine.execute(f"DELETE FROM workload WHERE query_num <= {midpoint}")
