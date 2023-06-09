import argparse
import datetime
import signal
import sys
import time
import traceback

from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from nyoom.analyze import Analyze
from nyoom.config import Config

CHOICES = ["tskip", "optimizer"]

STOPPU = False


def stoppu(signal, frame):
    global STOPPU
    STOPPU = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=CHOICES, required=True)
    parser.add_argument("--tskip_wiggle_std", type=float, default=2.0)
    parser.add_argument("--tskip_wiggle_sampen", type=float, default=20)
    parser.add_argument("--optimizer_cutoff_pct", type=float, default=10.0)
    parser.add_argument("--optimizer_min_processed", type=float, default=0)
    args = parser.parse_args()

    print(args)

    gym_engine = create_engine(
        Config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with gym_engine.connect() as gym_conn:
        setup_sqls = [
            # "DROP TABLE IF EXISTS nyoom_results",
            "CREATE TABLE IF NOT EXISTS nyoom_results (id serial, ts timestamp, pid int, token int, plan text)",
        ]

        relname_reltuples_map = {}
        indexname_tablename_map = {}
        trainer_engine = create_engine(
            Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
        )
        with trainer_engine.connect() as trainer_conn:
            print("Building relname_reltuples_map.")
            relname_reltuples_map_sql = text(
                "SELECT nspname AS schemaname, relname, reltuples "
                "FROM pg_class C LEFT JOIN pg_namespace N ON (C.relnamespace = N.oid) "
                "WHERE nspname NOT IN ('pg_catalog', 'information_schema') AND relkind = 'r' "
                "ORDER BY reltuples DESC"
            )
            results = trainer_conn.execute(relname_reltuples_map_sql)
            for row in results:
                schemaname, relname, reltuples = row
                assert schemaname == "public"
                relname_reltuples_map[relname] = reltuples

            print("Building indexname_tablename_map.")
            indexname_tablename_map_sql = text("SELECT indexname, tablename FROM pg_indexes")
            results = trainer_conn.execute(indexname_tablename_map_sql)
            for row in results:
                indexname, tablename = row
                indexname_tablename_map[indexname] = tablename
        trainer_engine.dispose()

        installed = False
        max_id = None
        while not installed:
            try:
                print("Installing nyoom.")
                for sql in setup_sqls:
                    gym_conn.execute(text(sql))
                result = gym_conn.execute(text("SELECT max(id) FROM nyoom_results"))
                max_id = result.scalar_one()
                installed = True
                print("Installed nyoom.")
            except SQLAlchemyError as e:
                print(e)
                print("Install failed, trying again.")
                time.sleep(5)

        while True:
            try:
                trainer_engine = create_engine(
                    Config.TRAINER_PG_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
                )

                with trainer_engine.connect() as trainer_conn:
                    trainer_conn.execute(text("CREATE EXTENSION IF NOT EXISTS nyoom"))
                    # Dump a snapshot of current state.
                    trainer_conn.execute(text("SELECT * FROM nyoom_enqueue_dump()"))
                    # Read the snapshot.
                    nyoom_read_sql = text("SELECT now() AS ts, pid, token, plan FROM nyoom_read_all()")
                    nyoom_results, results = [], trainer_conn.execute(nyoom_read_sql)
                    if results.returns_rows:
                        insert_sql = text(
                            "INSERT INTO nyoom_results (ts, pid, token, plan) VALUES (:ts, :pid, :token, :plan)"
                        )
                        for row in results:
                            ts, pid, token, plan = row
                            nyoom_results.append((ts, pid, token, plan))
                            insert_data = {"ts": ts, "pid": pid, "token": token, "plan": plan}
                            gym_conn.execute(insert_sql, insert_data)
                    # Analyze the snapshot.
                    # # TODO(WAN): probably better to have this happen on a different thread / microservice.
                    # # TODO(WAN): don't need to read the result since we just obtained it.
                    # results_df = pd.read_sql_table("nyoom_results", gym_conn)
                    # TODO(WAN): pd.read_sql and pd.read_sql_table is cursed for some reason.
                    active = [(pid, token) for _, pid, token, _ in nyoom_results]

                    for active_pid, active_token in active:
                        select_sql = text(
                            f"""
                            SELECT ts, pid, token, plan FROM nyoom_results
                            WHERE pid = {active_pid} AND token={token} AND id>{max_id}
                            ORDER BY id DESC
                            LIMIT 3
                            """
                        )
                        active_results = gym_conn.execute(select_sql)
                        active_results = [(ts, pid, token, plan) for ts, pid, token, plan in active_results]

                        analyzes = []
                        for ts, pid, token, plan in active_results:
                            try:
                                analyze = Analyze(relname_reltuples_map, indexname_tablename_map, plan)
                                try:
                                    # TODO(WAN): We no longer use these bounds.
                                    # analyze.compute_bounds()
                                    analyzes.append(analyze)
                                except Exception as e:
                                    filename = f"/nyoom/{pid}-{ts}.png"
                                    analyze.viz(filename)
                                    print(f"Error computing bounds, see: {filename}\n")
                                    with open(f"/nyoom/{pid}-{ts}-traceback.txt", "w") as f:
                                        traceback.print_exc(file=f)
                                    with open(f"/nyoom/{pid}-{ts}-plan.json", "w") as f:
                                        print(plan, file=f)
                            except:
                                continue

                        victim_plan_node_ids = None
                        if args.method == "tskip":
                            if len(analyzes) < 2:
                                # Not enough data to make a decision.
                                continue

                            analysis = Analyze.compare(
                                analyzes[-2], analyzes[-1],
                                wiggle_std=args.tskip_wiggle_std,
                                wiggle_sampen=args.tskip_wiggle_sampen,
                            )
                            victim_plan_node_ids = analysis["Stop These Plan Nodes"]
                        elif args.method == "optimizer":
                            if len(analyzes) < 1:
                                continue
                            victim_plan_node_ids = analyzes[-1].get_victims(
                                cutoff_pct=args.optimizer_cutoff_pct,
                                min_processed=args.optimizer_min_processed,
                            )
                        assert victim_plan_node_ids is not None

                        print(
                            f"{datetime.datetime.now()} Stopping {pid=} {token=}: ",
                            victim_plan_node_ids,
                        )
                        for plan_node_id in victim_plan_node_ids:
                            zw_sql = text(f"SELECT * FROM nyoom_enqueue_zw({pid}, {token}, {plan_node_id})")
                            trainer_conn.execute(zw_sql)
                    time.sleep(1)

                    if STOPPU:
                        trainer_conn.close()
                        sys.exit(0)
            except SQLAlchemyError as e:
                print(e)
            time.sleep(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, stoppu)
    main()
