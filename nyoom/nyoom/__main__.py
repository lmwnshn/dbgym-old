import time

from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from nyoom.analyze import Analyze
from nyoom.config import Config


def main():
    gym_engine = create_engine(
        Config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool, execution_options={"isolation_level": "AUTOCOMMIT"}
    )
    with gym_engine.connect() as gym_conn:
        setup_sqls = [
            "DROP TABLE IF EXISTS nyoom_results",
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
        while not installed:
            try:
                print("Installing nyoom.")
                for sql in setup_sqls:
                    gym_conn.execute(text(sql))
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

                while True:
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

                        # print("Analyzing: ", [pid for _, pid, _, _ in nyoom_results])
                        for ts, pid, token, plan in nyoom_results:
                            analyze = Analyze(relname_reltuples_map, indexname_tablename_map, plan)
                            try:
                                analyze.compute_bounds()
                            except Exception as e:
                                filename = f"/nyoom/{pid}-{ts}.png"
                                analyze.viz(filename)
                                print(e)
                                print(f"Error computing bounds, see: {filename}\n")

                            cutoff_pct = 10
                            min_processed = 1000
                            victim_plan_node_ids = analyze.get_victims(
                                cutoff_pct=cutoff_pct, min_processed=min_processed
                            )
                            print(
                                f"Stopping {pid=} {token=} [{cutoff_pct=}, {min_processed=}]: ",
                                victim_plan_node_ids,
                            )
                            for plan_node_id in victim_plan_node_ids:
                                zw_sql = text(f"SELECT * FROM nyoom_enqueue_zw({pid}, {token}, {plan_node_id})")
                                trainer_conn.execute(zw_sql)
                        time.sleep(1)
            except SQLAlchemyError as e:
                print(e)
            time.sleep(1)


if __name__ == "__main__":
    main()
