#!/usr/bin/env bash

echo "Creating volume 'tpch_queries'."

set -euxo pipefail

ROOT_DIR=$(pwd)
TPCH_SEED_START=15721
TPCH_SEED_END=16720

if [ ! -d "./build/tpch-kit" ]; then
  mkdir -p ./build
  cd ./build
  git clone git@github.com:lmwnshn/tpch-kit.git --single-branch --branch master --depth 1
  cd ./tpch-kit/dbgen
  make MACHINE=LINUX DATABASE=POSTGRESQL
  cd "${ROOT_DIR}"
fi

cd ./build/tpch-kit/dbgen
mkdir -p ./generated_queries
set +x
for seed in $(seq ${TPCH_SEED_START} ${TPCH_SEED_END}); do
  if [ ! -d "./generated_queries/${seed}" ]; then
    mkdir -p "./generated_queries/${seed}"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "./generated_queries/${seed}/${qnum}.sql"
    done
  fi
done
set -x
cd "${ROOT_DIR}"

docker run --volume=tpch_queries:/tpch_queries --name tpch_queries busybox true
docker cp --quiet "./build/tpch-kit/dbgen/generated_queries/." tpch_queries:/tpch_queries
docker rm tpch_queries

