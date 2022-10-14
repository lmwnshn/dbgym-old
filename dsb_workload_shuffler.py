import argparse
from pathlib import Path
from random import Random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", required=True, help="Comma-separated list of workloads to run.")
    parser.add_argument("--root_workload_dir", default="./artifact/dsb/workload", help="Root directory of all workloads.")
    parser.add_argument("--seed", default=15721, help="Random seed.")
    args = parser.parse_args()

    rng = Random(args.seed)

    root_workload_dir = Path(args.root_workload_dir)
    workload_folders = [root_workload_dir / workload for workload in args.workloads.split(",")]

    assert all([w.exists() and w.is_dir() for w in workload_folders]), "Workloads must exist."
    sqls = []
    for workload_folder in workload_folders:
        sqls.extend(workload_folder.rglob("*.sql"))
    rng.shuffle(sqls)

    for sql_file in sqls:
        print(f"{sql_file.absolute()}")


if __name__ == "__main__":
    main()