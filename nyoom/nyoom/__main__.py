from nyoom.config import Config
from nyoom.analyze import Analyze

import time

from sqlalchemy import NullPool, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def main():
    gym_engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, poolclass=NullPool,
                               execution_options={"isolation_level": "AUTOCOMMIT"})
    with gym_engine.connect() as gym_conn:
        setup_sqls = [
            "DROP TABLE IF EXISTS nyoom_results",
            "CREATE TABLE IF NOT EXISTS nyoom_results (id serial, ts timestamp, pid int, token int, plan text)",
            "DROP TABLE IF EXISTS nyoom_signal",
            "CREATE TABLE IF NOT EXISTS nyoom_signal (run boolean)",
            "INSERT INTO nyoom_signal VALUES (FALSE)",
        ]

        for sql in setup_sqls:
            gym_conn.execute(text(sql))

        while True:
            result = gym_conn.execute(text("SELECT run FROM nyoom_signal")).fetchone()
            if result[0]:
                try:
                    trainer_engine = create_engine(Config.TRAINER_PG_URI, poolclass=NullPool,
                                                   execution_options={"isolation_level": "AUTOCOMMIT"})
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

                            relname_reltuples_map_sql = text(
                                "SELECT nspname AS schemaname, relname, reltuples "
                                "FROM pg_class C LEFT JOIN pg_namespace N ON (C.relnamespace = N.oid) "
                                "WHERE nspname NOT IN ('pg_catalog', 'information_schema') AND relkind = 'r' "
                                "ORDER BY reltuples DESC"
                            )
                            results = trainer_conn.execute(relname_reltuples_map_sql)
                            relname_reltuples_map = {}
                            for row in results:
                                schemaname, relname, reltuples = row
                                assert schemaname == "public"
                                relname_reltuples_map[relname] = reltuples

                            indexname_tablename_map_sql = text("SELECT indexname, tablename FROM pg_indexes")
                            results = trainer_conn.execute(indexname_tablename_map_sql)
                            indexname_tablename_map = {}
                            for row in results:
                                indexname, tablename = row
                                indexname_tablename_map[indexname] = tablename

                            for ts, pid, token, plan in nyoom_results:
                                analyze = Analyze(relname_reltuples_map, indexname_tablename_map, plan)
                                victim_plan_node_ids = analyze.get_victims(cutoff_pct=10, min_processed=0)
                                if len(victim_plan_node_ids) > 0:
                                    analyze.viz(f"/nyoom/{pid}-{ts}.png")
                                for plan_node_id in victim_plan_node_ids:
                                    zw_sql = text(f"SELECT * FROM nyoom_enqueue_zw({pid}, {token}, {plan_node_id})")
                                    print("Sending: ", zw_sql)
                                    trainer_conn.execute(zw_sql)
                            time.sleep(1)
                except SQLAlchemyError as e:
                    print(e)
                    pass
            time.sleep(1)


if __name__ == "__main__":
    main()
