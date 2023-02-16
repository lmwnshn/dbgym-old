#!/usr/bin/env bash

source .env

set -euxo pipefail

./setup/tpch_sf1.sh
./setup/tpch_schema.sh
./setup/tpch_queries.sh

docker compose down
docker compose up --build --detach --wait
docker compose run --build dbgym
