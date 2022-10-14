#!/bin/env bash

set -e
set -u
set -x
set -o pipefail

DB_NAME="prod_db"
DB_USER="prod_user"
DB_PASS="prod_pass"
ROOT_WD=$(pwd)

# User setup.
sudo -u postgres --login psql -c "drop database if exists ${DB_NAME}"
sudo -u postgres --login psql -c "drop user if exists ${DB_USER}"
sudo -u postgres --login psql -c "create user ${DB_USER} with superuser encrypted password '${DB_PASS}'"

rm -rf ./artifact
rm -rf ./build

mkdir -p ./artifact
mkdir -p ./artifact/dsb
mkdir -p ./artifact/prod_dbms
mkdir -p ./build

# Clone DSB if necessary.
if [ ! -d './build/dsb' ]; then
  git clone git@github.com:lmwnshn/dsb.git --single-branch --branch feature_generate_workload_linux --depth=1 ./build/dsb
  cd ./build/dsb/code/tools
  make
  cd "${ROOT_WD}"
fi

# Setup benchmark.
# Workload config.
cat > ./artifact/dsb/workload_config.json << EOF
{
  "output_dir": "${ROOT_WD}/artifact/dsb/workload/",
  "binary_dir": "${ROOT_WD}/build/dsb/code/tools",
  "query_template_root_dir": "${ROOT_WD}/build/dsb/query_templates_pg/",
  "dialect" : "postgres",
  "workload":
  [
    {
      "id" : "train_default",
      "query_template_names" : [],
      "instance_count" : 200,
      "param_dist" : "default",
      "rngseed" : 15721
    },
    {
      "id" : "train_gaussian",
      "query_template_names" : [],
      "instance_count" : 200,
      "param_dist" : "normal",
      "param_sigma" : 2,
      "param_center" : 0,
      "rngseed" : 15721
    },
    {
      "id" : "test_default",
      "query_template_names" : [],
      "instance_count" : 20,
      "param_dist" : "default",
      "rngseed" : 15445
    }
  ]
}
EOF
# Data generation.
if [ ! -d "${ROOT_WD}/artifact/dsb/data" ]; then
  mkdir -p "${ROOT_WD}/artifact/dsb/data"
  cd ./build/dsb/code/tools
  ./dsdgen -scale 1 -terminate n -force -rngseed 15721 -dir "${ROOT_WD}/artifact/dsb/data"
  cd "${ROOT_WD}"
fi
# Workload generation.
if [ ! -d "${ROOT_WD}/artifact/dsb/workload" ]; then
  cd ./build/dsb/code/tools
  python3 ../../scripts/generate_workload.py --workload_config_file "${ROOT_WD}/artifact/dsb/workload_config.json" --os linux
  cd "${ROOT_WD}"

  # These queries are really slow.
  rm -rf ./artifact/dsb/workload/train_default/query{001,014,032,072,072_spj,081,092}
  rm -rf ./artifact/dsb/workload/train_gaussian/query{001,014,032,072,072_spj,081,092}
  rm -rf ./artifact/dsb/workload/test_default/query{001,014,032,072,072_spj,081,092}
fi

#query 32: timeout
#query 81: 58s
#query 72: 12s
#query 92: 7s (edited


# Database setup.
PGPASSWORD=${DB_PASS} dropdb --host=localhost --username=${DB_USER} --if-exists ${DB_NAME}
PGPASSWORD=${DB_PASS} createdb --host=localhost --username=${DB_USER} ${DB_NAME}

# pgtune setup.
# DB Version: 14
# OS Type: linux
# DB Type: mixed
# Total Memory (RAM): 48 GB
# Data Storage: ssd
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET max_connections = '100'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET shared_buffers = '12GB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET effective_cache_size = '36GB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET maintenance_work_mem = '2GB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET checkpoint_completion_target = '0.9'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET wal_buffers = '16MB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET default_statistics_target = '100'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET random_page_cost = '1.1'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET effective_io_concurrency = '200'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET work_mem = '31457kB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET min_wal_size = '1GB'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET max_wal_size = '4GB'"
sudo systemctl restart postgresql
until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

# Load the benchmark.
cd ./build/dsb/scripts
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --file=./create_tables.sql
set +x
for datfile in "${ROOT_WD}"/artifact/dsb/data/*.dat; do
  table_name=$(basename ${datfile} | sed -e 's/\.dat$//')
  csv_file=${datfile}
  set -x
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="\timing" --command="TRUNCATE ${table_name}"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="\timing" --command="\COPY ${table_name} FROM '${csv_file}' WITH (DELIMITER '|', FORMAT CSV)"
  set +x
done
set -x
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="\timing" --file=./dsb_index_pg.sql
cd -

# Save the state.
# We want the state _before_ the workload is run.
PGPASSWORD=${DB_PASS} pg_dump --host=localhost --username=${DB_USER} --format=directory --file=./artifact/prod_dbms/state ${DB_NAME}

_run_workloads() {
  workloads="${1}"
  save_location="${2}"

  echo "Running workloads: ${workloads}, saving to ${save_location}".
  echo "$(date)"

  # Clear the log folder.
  sudo bash -c "rm -rf /var/lib/postgresql/14/main/log/*"

  # Enable logging.
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_destination='csvlog'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET logging_collector='on'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_statement='all'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_connections='on'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_disconnections='on'"
  sudo systemctl restart postgresql
  until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

  # Run the benchmark.
  for sql_file in $(python3 dsb_workload_shuffler.py --workloads ${workloads} --seed 15721); do
    PGPASSWORD=${DB_PASS} psql -P pager=off --host=localhost --dbname=${DB_NAME} --command="\timing" --command="set statement_timeout='1min'" --username=${DB_USER} --file=${sql_file}
  done

  # Disable logging.
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_destination='stderr'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET logging_collector='off'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_statement='none'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_connections='off'"
  PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_disconnections='off'"
  sudo systemctl restart postgresql
  until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

  # Save the workload.
  sudo bash -c "cat /var/lib/postgresql/14/main/log/*.csv > ${save_location}"

  echo "$(date)"
}

#set +x
_run_workloads "train_default" "./artifact/prod_dbms/train_default_workload.csv" > train_default.txt 2>&1
_run_workloads "train_gaussian" "./artifact/prod_dbms/train_gaussian_workload.csv" > train_gaussian.txt 2>&1
_run_workloads "test_default" "./artifact/prod_dbms/test_workload.csv" > test_default.txt 2>&1
