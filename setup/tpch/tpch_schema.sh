#!/usr/bin/env bash

if docker volume inspect tpch_schema; then
  echo "Volume 'tpch_schema' exists."
  exit
fi

echo "Volume 'tpch_schema' doesn't exist, creating."

set -euxo pipefail

docker run --volume=tpch_schema:/tpch_schema --name tpch_schema busybox true
docker cp --quiet ./setup/tpch/tpch_schema.sql tpch_schema:/tpch_schema
docker cp --quiet ./setup/tpch/tpch_constraints.sql tpch_schema:/tpch_schema
docker rm tpch_schema
