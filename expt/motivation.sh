#!/bin/bash

set -euxo pipefail

BUILD_DIR="$(pwd)/build/postgres/"
BIN_DIR="$(pwd)/build/postgres/bin/"
POSTGRES_USER="noisepage_user"
POSTGRES_PASSWORD="noisepage_pass"
POSTGRES_DB="noisepage_db"
POSTGRES_PORT=15799

ROOT_DIR=$(pwd)
TPCH_DATA="${ROOT_DIR}/artifact/tpch-kit/data"
TPCH_QUERIES="${ROOT_DIR}/artifact/tpch-kit/queries"

mkdir -p ./artifact
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
  ./dbgen -vf -s 1
  # cp dss.ddl "${TPCH_DATA}/dss.ddl"
  # Use the BenchBase schema.
  cp "${ROOT_DIR}/cmudb/expt/wan/motivation/tpch_schema.sql" "${TPCH_DATA}/schema.sql"
  mv ./*.tbl "${TPCH_DATA}"
  cd "${ROOT_DIR}"
fi

if [ ! -d "${TPCH_QUERIES}" ]; then
  mkdir -p "${TPCH_QUERIES}"
  cd ./build/tpch-kit/dbgen
  for seed in {15721..15723}; do
    mkdir -p "${TPCH_QUERIES}/${seed}/"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "${TPCH_QUERIES}/${seed}/${qnum}.sql"
    done
  done
  cd "${ROOT_DIR}"
fi

PGPASSWORD=${POSTGRES_PASSWORD} ${BIN_DIR}/psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
  -f "${TPCH_DATA}/schema.sql"
tpch_tables=("region" "nation" "part" "supplier" "partsupp" "customer" "orders" "lineitem")
for table_name in "${tpch_tables[@]}"; do
  PGPASSWORD=${POSTGRES_PASSWORD} ${BIN_DIR}/psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
    -c "TRUNCATE ${table_name} CASCADE"
done
for table_name in "${tpch_tables[@]}"; do
  table_path="${ROOT_DIR}/artifact/tpch-kit/data/${table_name}.tbl"
  PGPASSWORD=${POSTGRES_PASSWORD} ${BIN_DIR}/psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -p ${POSTGRES_PORT} \
    -c "\\COPY ${table_name} FROM '${table_path}' CSV DELIMITER '|'"
done

for seed in {15721..15723}; do
  for qnum in {1..2}; do
    "${TPCH_QUERIES}/${seed}/${qnum}.sql"
  done
done