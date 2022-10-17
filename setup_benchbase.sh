#!/bin/env bash

set -e
set -u
set -x
set -o pipefail

DB_NAME="prod_db"
DB_USER="prod_user"
DB_PASS="prod_pass"
BENCHMARK="tpch"
ROOT_WD=$(pwd)

# # User setup.
# sudo -u postgres --login psql -c "drop database if exists ${DB_NAME}"
# sudo -u postgres --login psql -c "drop user if exists ${DB_USER}"
# sudo -u postgres --login psql -c "create user ${DB_USER} with superuser encrypted password '${DB_PASS}'"
# 
# rm -rf ./artifact
# rm -rf ./build
# 
# mkdir -p ./artifact
# mkdir -p ./artifact/benchbase
# mkdir -p ./artifact/prod_dbms
# mkdir -p ./build
# 
# # Clone BenchBase if necessary.
# if [ ! -d './build/benchbase' ]; then
#   git clone git@github.com:cmu-db/benchbase.git --single-branch --branch main --depth=1 ./build/benchbase
#   cd ./build/benchbase
#   ./mvnw clean package -P postgres -Dmaven.test.skip=true
#   cd ./target
#   tar xvzf benchbase-postgres.tgz
#   cd ./benchbase-postgres
#   cd "${ROOT_WD}"
# fi
# 
# # Set benchmark XML.
cp "./build/benchbase/target/benchbase-postgres/config/postgres/sample_${BENCHMARK}_config.xml" "${ROOT_WD}/artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --update '/parameters/url' --value "jdbc:postgresql://localhost:5432/${DB_NAME}?preferQueryMode=extended" "./artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --update '/parameters/username' --value "${DB_USER}" "./artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --update '/parameters/password' --value "${DB_PASS}" "./artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --update '/parameters/isolation' --value "TRANSACTION_READ_COMMITTED" "./artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --update '/parameters/scalefactor' --value "1" "./artifact/benchbase/${BENCHMARK}_config.xml"
# # TPCC
# #xmlstarlet edit --inplace --update '/parameters/works/work/rate' --value "unlimited" "./artifact/benchbase/${BENCHMARK}_config.xml"
# #xmlstarlet edit --inplace --update '/parameters/works/work/time' --value "60" "./artifact/benchbase/${BENCHMARK}_config.xml"
# # TPCH
set +x
xmlstarlet edit --inplace --delete '/parameters/works/work/time' "./artifact/benchbase/${BENCHMARK}_config.xml"
xmlstarlet edit --inplace --delete 'parameters/works/work' "./artifact/benchbase/${BENCHMARK}_config.xml"
for i in {1..200}
do
  xmlstarlet edit --inplace --omit-decl \
  --subnode '/parameters/works' --type elem -n "work" \
  --subnode "/parameters/works/work[${i}]" --type elem -n "serial" -v "true" \
  --subnode "/parameters/works/work[${i}]" --type elem -n "rate" -v "unlimited" \
  --subnode "/parameters/works/work[${i}]" --type elem -n "weights" -v "1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1" \
  "./artifact/benchbase/${BENCHMARK}_config.xml"
done
set -x
# 
# # Database setup.
# PGPASSWORD=${DB_PASS} dropdb --host=localhost --username=${DB_USER} --if-exists ${DB_NAME}
# PGPASSWORD=${DB_PASS} createdb --host=localhost --username=${DB_USER} ${DB_NAME}
# 
# # pgtune setup.
# # DB Version: 14
# # OS Type: linux
# # DB Type: mixed
# # Total Memory (RAM): 48 GB
# # Data Storage: ssd
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET max_connections = '100'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET shared_buffers = '12GB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET effective_cache_size = '36GB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET maintenance_work_mem = '2GB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET checkpoint_completion_target = '0.9'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET wal_buffers = '16MB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET default_statistics_target = '100'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET random_page_cost = '1.1'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET effective_io_concurrency = '200'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET work_mem = '31457kB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET min_wal_size = '1GB'"
# PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET max_wal_size = '4GB'"
# sudo systemctl restart postgresql
# until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done
# 
# # Load the benchmark.
# cd ./build/benchbase/target/benchbase-postgres
# java -jar benchbase.jar -b "${BENCHMARK}" -c "${ROOT_WD}/artifact/benchbase/${BENCHMARK}_config.xml" --create=true --load=true
# cd -
# 
# # Save the state.
# # We want the state _before_ the workload is run.
# PGPASSWORD=${DB_PASS} pg_dump --host=localhost --username=${DB_USER} --format=directory --file=./artifact/prod_dbms/state ${DB_NAME}

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
cd ./build/benchbase/target/benchbase-postgres
java -jar benchbase.jar -b "${BENCHMARK}" -c "${ROOT_WD}/artifact/benchbase/${BENCHMARK}_config.xml" --execute=true
cd -

# Disable logging.
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_destination='stderr'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET logging_collector='off'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_statement='none'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_connections='off'"
PGPASSWORD=${DB_PASS} psql --host=localhost --dbname=${DB_NAME} --username=${DB_USER} --command="ALTER SYSTEM SET log_disconnections='off'"
sudo systemctl restart postgresql
until PGPASSWORD=${DB_PASS} pg_isready --host=localhost --dbname=${DB_NAME} --username=${DB_USER} ; do sleep 1 ; done

# Save the workload.
sudo bash -c "cat /var/lib/postgresql/14/main/log/*.csv > ./artifact/prod_dbms/workload.csv"
