"""
Convert PostgreSQL query log to workload.db.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pglast
from sqlalchemy import Column, Integer, String, Table, create_engine, insert, text
from tqdm import tqdm

from dbgym.util.sql import substitute
from dbgym.util.workload_schema import get_workload_schema

_PG_LOG_DTYPES = {
    "log_time": str,
    "user_name": str,
    "database_name": str,
    "process_id": "Int64",
    "connection_from": str,
    "session_id": str,
    "session_line_num": "Int64",
    "command_tag": str,
    "session_start_time": str,
    "virtual_transaction_id": str,
    "transaction_id": "Int64",
    "error_severity": str,
    "sql_state_code": str,
    "message": str,
    "detail": str,
    "hint": str,
    "internal_query": str,
    "internal_query_pos": "Int64",
    "context": str,
    "query": str,
    "query_pos": "Int64",
    "location": str,
    "application_name": str,
    # PostgreSQL 13+.
    "backend_type": str,
    # PostgreSQL 14+.
    "leader_pid": "Int64",
    "query_id": "Int64",
}


def _read_postgresql_csvlog(pg_csvlog):
    df = pd.read_csv(
        pg_csvlog,
        names=_PG_LOG_DTYPES.keys(),
        parse_dates=["log_time"],
        usecols=[
            "log_time",
            "message",
            "detail",
        ],
        dtype=_PG_LOG_DTYPES,
        header=None,
    )

    simple = r"^statement: ([\s\S]*)"
    extended = r"^execute .+: ([\s\S]*)"
    regex = f"(?:{simple})|(?:{extended})"

    with tqdm(total=5, desc="Processing CSVLOG into DataFrame.", leave=False) as pbar:
        with tqdm(desc="Extracting query templates.", leave=False):
            query = df["message"].str.extract(regex, flags=re.IGNORECASE)
            # Combine the capture groups for simple and extended query protocol.
            query = query[0].fillna(query[1])
            # print("TODO(WAN): Disabled SQL format for being too slow.")
            # Prettify each SQL query for standardized formatting.
            # query = query.apply(pglast.prettify, na_action='ignore')
            df["query_raw"] = query
        pbar.update(1)

        with tqdm(desc="Extracting parameters.", leave=False):
            df["params"] = df["detail"].apply(_extract_params)
        pbar.update(1)

        with tqdm(desc="Substituting parameters back into query.", leave=False):
            # TODO(WAN): You _could_ do the hacky thing for queries without $.
            df["query_subst"] = df[["query_raw", "params"]].apply(_substitute_row, axis=1)
            df = df.drop(columns=["query_raw", "params"])
        pbar.update(1)

        with tqdm(desc="Re-parsing query.", leave=False):
            template_param = df["query_subst"].apply(_parse)
            df = df.assign(
                query_template=template_param.map(lambda x: x[0]),
                query_params=template_param.map(lambda x: x[1]),
            )
            df = df[df["query_template"].astype(bool)].copy().reset_index()
        pbar.update(1)

        with tqdm(desc="Adding minor changes.", leave=False):
            df["query_template"] = df["query_template"].astype("category")
            start_time = df.iloc[0]["log_time"]
            df["elapsed_s"] = (df["log_time"] - start_time).dt.total_seconds()

            stored_columns = {
                "elapsed_s",
                "query_template",
                "query_params",
            }
            df = df.drop(columns=set(df.columns) - stored_columns)
        pbar.update(1)

    return df


def _extract_params(detail):
    detail = str(detail)
    prefix = "parameters: "
    idx = detail.find(prefix)
    if idx == -1:
        return {}
    parameter_list = detail[idx + len(prefix) :]
    params = {}
    for pstr in parameter_list.split(", "):
        pnum, pval = pstr.split(" = ")
        assert pnum.startswith("$")
        assert pnum[1:].isdigit()
        params[pnum] = pval
    return params


def _substitute_row(row):
    query, params = row["query_raw"], row["params"]
    if query is pd.NA or query is np.nan:
        return pd.NA
    return substitute(str(query), params, onerror="ignore")


def _parse(sql):
    if sql is pd.NA or sql is np.nan:
        return "", ()
    sql = str(sql)
    new_sql, params, last_end = [], [], 0
    for token in pglast.parser.scan(sql):
        token_str = str(sql[token.start : token.end + 1])
        if token.start > last_end:
            new_sql.append(" ")
        if token.name in ["ICONST", "FCONST", "SCONST"]:
            # Integer, float, or string constant.
            new_sql.append("$" + str(len(params) + 1))
            # HACK: See if you can steal a unary minus from the current template being built up.
            # Skip index -1 because that's going to be a $1 $2 etc. parameter.
            if new_sql[-2] == "-":
                # Can't replace, in case of "a=b-1" predicates.
                new_sql[-2] = "+"
                token_str = f"-{token_str}"
            # # Quote for consistency.
            # if token_str[0] != "'" and token_str[-1] != "'":
            #     token_str = f"'{token_str}'"
            params.append(token_str)
        else:
            new_sql.append(token_str)
        last_end = token.end + 1
    new_sql = "".join(new_sql)
    return new_sql, tuple(params)


def convert_postgresql_csvlog_to_workload(postgresql_csvlog_path: Path, save_path: Path):
    assert postgresql_csvlog_path.suffix == ".csv", f"CSVLOG format? {postgresql_csvlog_path}"

    Path(save_path).unlink(missing_ok=True)
    workload_id = 1
    engine = create_engine(f"sqlite:///{save_path}")
    metadata = get_workload_schema()
    metadata.create_all(engine)

    df = _read_postgresql_csvlog(postgresql_csvlog_path)
    engine.execute("PRAGMA journal_mode = MEMORY")
    engine.execute("PRAGMA synchronous = OFF")
    insert_batch = {}

    def try_insert(key, val=None, batch_threshold=5000):
        nonlocal insert_batch
        if val is not None:
            insert_batch[key] = insert_batch.get(key, [])
            insert_batch[key].append(val)
        values = insert_batch[key]
        if len(values) >= batch_threshold:
            engine.execute(metadata.tables[key].insert().values(values))
            del insert_batch[key]

    categories = df["query_template"].dtype.categories
    templates = {query_template: template_id for template_id, query_template in enumerate(categories, 1)}
    for query_template, template_id in tqdm(templates.items(), total=len(templates), desc="Writing out templates."):
        try_insert("query_template", (template_id, query_template))
        num_params = len(df[df["query_template"] == query_template].iloc[0]["query_params"])
        if num_params > 0:
            params = [Column(f"param_{i}", String) for i in range(1, num_params + 1)]
            query_template_table = Table(
                f"template_{template_id}_params",
                metadata,
                Column("id", Integer, primary_key=True),
                *params,
            )
            query_template_table.create(engine)
    try_insert("query_template", batch_threshold=0)

    template_params_map = {}
    for query_num, (query_template, query_params, elapsed_s) in enumerate(
        tqdm(df.itertuples(index=False), total=len(df), desc="Writing out params."), 1
    ):
        template_id = templates[query_template]

        params_id = None
        if len(query_params) > 0:
            params_tup = tuple(query_params)
            if template_id in template_params_map and params_tup in template_params_map[template_id]:
                params_id = template_params_map[template_id][params_tup]
            else:
                template_params_map[template_id] = template_params_map.get(template_id, {})
                template_params_map[template_id][params_tup] = len(template_params_map[template_id]) + 1
                params_id = template_params_map[template_id][params_tup]
                params_table = f"template_{template_id}_params"
                try_insert(params_table, (params_id, *query_params))

        try_insert("workload", (workload_id, query_num, elapsed_s, template_id, params_id))

    keys = list(insert_batch.keys())
    for key in keys:
        try_insert(key, batch_threshold=0)


def convert_sqls_to_postgresql_csvlog(sqls: list[str], save_path: Path):
    # TODO(WAN): Obviously, very incomplete.
    start_ts = pd.Timestamp.now(tz="UTC")
    current_ts = start_ts
    user_name = "noisepage_user"
    database_name = "noisepage_pass"
    process_id = 15721
    connection_from = "127.0.0.1:50721"
    session_id = f"{int(start_ts.timestamp()):x}.{process_id:x}"
    session_line_num = 1
    command_tag = "idle"
    session_start_time = start_ts
    virtual_transaction_id_backend_number = 1
    virtual_transaction_id_counter = 0
    transaction_id = 0
    error_severity = "LOG"
    sql_state_code = "00000"
    detail = ""
    hint = ""
    internal_query = ""
    internal_query_pos = ""
    context = ""
    query = ""
    query_pos = ""
    location = ""
    application_name = "psql"
    backend_type = "client backend"
    leader_pid = ""
    query_id = 0

    def quote(s: str):
        return f'"{s}"'

    def format_ts(ts: pd.Timestamp):
        tz_format = "%Y-%m-%d %H:%M:%S.%f"
        time = ts.strftime(tz_format)[:-3]
        timezone = ts.strftime("%Z")
        return f"{time} {timezone}"

    with open(save_path, "w") as csvfile:
        for sql in sqls:
            current_ts += pd.Timedelta(seconds=1)
            assert '"' not in sql, "Bad SQL query?"
            message = f"statement: {sql}"
            session_line_num += 1
            virtual_transaction_id_counter += 1
            virtual_transaction_id = f"{virtual_transaction_id_backend_number}/{virtual_transaction_id_counter}"

            output = (
                f"{format_ts(current_ts)},"
                f"{quote(user_name)},"
                f"{quote(database_name)},"
                f"{process_id},"
                f"{quote(connection_from)},"
                f"{session_id},"
                f"{session_line_num},"
                f"{quote(command_tag)},"
                f"{format_ts(session_start_time)},"
                f"{virtual_transaction_id},"
                f"{transaction_id},"
                f"{error_severity},"
                f"{sql_state_code},"
                f"{quote(message)},"
                f"{detail},"
                f"{hint},"
                f"{internal_query},"
                f"{internal_query_pos},"
                f"{context},"
                f"{query},"
                f"{query_pos},"
                f"{location},"
                f"{quote(application_name)},"
                f"{quote(backend_type)},"
                f"{leader_pid},"
                f"{query_id}"
            )
            print(output, file=csvfile)
