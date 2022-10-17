"""
Defines the format of workload.db.
"""

from sqlalchemy import Column, ForeignKey, Index, Integer, MetaData, String, Table


def get_workload_schema() -> MetaData:
    metadata = MetaData()

    query_template = Table(
        "query_template",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("template", String, nullable=False),
    )

    workload = Table(
        "workload",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("query_num", Integer, primary_key=True),
        Column("elapsed_s", Integer, nullable=False),
        Column("template_id", Integer, ForeignKey("query_template.id"), nullable=False),
        Column("params_id", Integer, nullable=True),
        Index("idx_elapsed_s", "elapsed_s"),
    )

    return metadata
