from typing import TypedDict

import sqlalchemy

from dbgym.envs.state import State
from dbgym.envs.workload import Workload

Stats = TypedDict("Stats", {"min": float, "median": float, "max": float})
Column = TypedDict("Column", {"name": str, "stats": Stats}, total=False)
Index = TypedDict(
    "Index", {"name": str, "column_names": list[str], "unique": bool}, total=False
)
TableAttr = TypedDict(
    "TableAttr", {"columns": list[Column], "indexes": list[Index]}, total=False
)
Tables = dict[str, TableAttr]  # Tables = table name -> table attributes
Schema = dict[str, Tables]  # Schema = schema name -> tables in the Schema
Snapshot = TypedDict("Snapshot", {"schemas": Schema})


class GymSpec:
    def __init__(
        self,
        prod_db_connstr,
        historical_workload=None,
        historical_state=None,
        wan=False,
    ):
        """

        Parameters
        ----------
        prod_db_connstr : str
            Connection string to the production DBMS.
            Must be in SQLAlchemy format.
        """
        # Debug flag. Currently, speeds things up.
        self._wan: bool = wan

        self.prod_db_connstr: str = prod_db_connstr

        self.historical_workload: Workload = historical_workload
        self.historical_state: State = historical_state

        self._engine: sqlalchemy.engine.Engine = sqlalchemy.create_engine(
            self.prod_db_connstr
        )
        self._inspector: sqlalchemy.engine.Inspector = sqlalchemy.inspect(self._engine)

        self.snapshot: Snapshot = self._snapshot_db()

        self.schema_summary = [
            (schema_name, table_name, self.snapshot["schemas"][schema_name][table_name])
            for schema_name in self.snapshot["schemas"]
            for table_name in self.snapshot["schemas"][schema_name]
        ]

    def _snapshot_db(self):
        """Take a snapshot of the current DB state."""
        schemas: Schema = {}
        for schema_name in self._inspector.get_schema_names():
            if schema_name == "information_schema":
                # We're not tuning this.
                continue
            schemas[schema_name]: Tables = {}
            for (table_name, fkcs) in self._inspector.get_sorted_table_and_fkc_names(
                schema_name
            ):
                if table_name is None:
                    # The last iteration consists of FKs that would require a separate CREATE statement.
                    continue
                table_attr: TableAttr = {}
                table_attr["columns"]: list[Column] = self._inspector.get_columns(
                    table_name, schema_name
                )

                # Unfortunately, get_indexes() doesn't include pkey.
                pkey = self._inspector.get_pk_constraint(table_name, schema_name)
                pkey["column_names"] = pkey["constrained_columns"]
                del pkey["constrained_columns"]
                pkey["unique"] = True
                indexes = [pkey] + self._inspector.get_indexes(table_name, schema_name)
                table_attr["indexes"]: list[Index] = indexes

                column_names = [column["name"] for column in table_attr["columns"]]

                if self._wan:
                    select_targets = ", ".join(
                        [
                            f"min({column_name}), "
                            f"min({column_name}), "  # min is in stats and fast, median isn't.
                            f"max({column_name})"
                            for column_name in column_names
                        ]
                    )
                else:
                    select_targets = ", ".join(
                        [
                            f"min({column_name}), "
                            f"percentile_disc(0.5) within group (order by {column_name}), "
                            f"max({column_name})"
                            for column_name in column_names
                        ]
                    )
                query = f"SELECT {select_targets} FROM {table_name}"
                attr_stats = self._engine.execute(query).fetchall()
                assert (
                    len(attr_stats) == 1
                ), f"Something weird has happened, check the query: {query}"
                attr_stats = attr_stats[0]
                attr_stats = [
                    attr_stats[i : i + 3] for i in range(0, len(attr_stats), 3)
                ]
                attr_stats = {
                    column_name: attr_stat
                    for column_name, attr_stat in zip(column_names, attr_stats)
                }
                for column in table_attr["columns"]:
                    column_name = column["name"]
                    assert (
                        "stats" not in table_attr["columns"]
                    ), f"`stats` name clash for {table_name}.{column_name}."
                    attr_min, attr_median, attr_max = attr_stats[column_name]
                    column["stats"]: Stats = {
                        "min": attr_min,
                        "median": attr_median,
                        "max": attr_max,
                    }

                schemas[schema_name][table_name]: TableAttr = table_attr

        snapshot: Snapshot = {
            "schemas": schemas,
        }
        return snapshot

    def __str__(self):
        return self.prod_db_connstr
