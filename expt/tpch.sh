#!/usr/bin/env bash

# Produces:
#   - ./artifact/prod_dbms/state
#   - ./artifact/tpch-kit/queries/{seed}/{1..22}.sql

set -euxo pipefail

POSTGRES_USER="prod_user"
POSTGRES_PASSWORD="prod_pass"
POSTGRES_DB="prod_db"
POSTGRES_PORT=5432

ROOT_DIR=$(pwd)
TPCH_SCHEMA="${ROOT_DIR}/expt/tpch_schema.sql"
TPCH_DATA="${ROOT_DIR}/artifact/tpch-kit/data"
TPCH_QUERIES="${ROOT_DIR}/artifact/tpch-kit/queries"
TPCH_SEED_START=15721
TPCH_SEED_END=16621

mkdir -p ./artifact/prod_dbms
mkdir -p ./build

if [ ! -d "./build/tpch-kit" ]; then
  cd ./build
  git clone git@github.com:lmwnshn/tpch-kit.git --single-branch --branch master --depth 1
  cd ./tpch-kit/dbgen
  make MACHINE=LINUX DATABASE=POSTGRESQL
  cd "${ROOT_DIR}"
fi

if [ ! -d "${TPCH_DATA}" ]; then
  mkdir -p "${TPCH_DATA}"
  cd ./build/tpch-kit/dbgen
  ./dbgen -vf -s 10
  # cp dss.ddl "${TPCH_DATA}/dss.ddl"
  # Use the BenchBase schema.
  cp "${TPCH_SCHEMA}" "${TPCH_DATA}/schema.sql"
  mv ./*.tbl "${TPCH_DATA}"
  cd "${ROOT_DIR}"
fi

if [ ! -d "${TPCH_QUERIES}" ]; then
  mkdir -p "${TPCH_QUERIES}"
  cd ./build/tpch-kit/dbgen
  set +x
  for seed in $(seq ${TPCH_SEED_START} ${TPCH_SEED_END}); do
    mkdir -p "${TPCH_QUERIES}/${seed}/"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "${TPCH_QUERIES}/${seed}/${qnum}.sql"
    done
  done
  set -x
  cd "${ROOT_DIR}"
fi

PGPASSWORD=${POSTGRES_PASSWORD} psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
  -f "${TPCH_DATA}/schema.sql"
tpch_tables=("region" "nation" "part" "supplier" "partsupp" "customer" "orders" "lineitem")
for table_name in "${tpch_tables[@]}"; do
  PGPASSWORD=${POSTGRES_PASSWORD} psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
    -c "TRUNCATE ${table_name} CASCADE"
done
for table_name in "${tpch_tables[@]}"; do
  table_path="${ROOT_DIR}/artifact/tpch-kit/data/${table_name}.tbl"
  PGPASSWORD=${POSTGRES_PASSWORD} psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
    -c "\\COPY ${table_name} FROM '${table_path}' CSV DELIMITER '|'"
done

rm -rf ./artifact/prod_dbms/state
PGPASSWORD=${POSTGRES_PASSWORD} pg_dump --host=localhost --username=${POSTGRES_USER} --format=directory --file=./artifact/prod_dbms/state ${POSTGRES_DB}
