#!/usr/bin/env bash

if docker volume inspect dsb_schema; then
  echo "Volume 'dsb_schema' exists."
  exit
fi

echo "Volume 'dsb_schema' doesn't exist, creating."

set -euxo pipefail

ROOT_DIR=$(pwd)

if [ ! -d "./build/dsb" ]; then
  mkdir -p ./build
  cd ./build
  git clone git@github.com:lmwnshn/dsb.git --single-branch --branch fix_ubuntu --depth 1
  cd ./dsb/code/tools
  make
  cd "${ROOT_DIR}"
fi

docker run --volume=dsb_schema:/dsb_schema --name dsb_schema busybox true
docker cp --quiet ./build/dsb/scripts/dsb_index_pg.sql dsb_schema:/dsb_schema
docker cp --quiet ./build/dsb/scripts/create_tables.sql dsb_schema:/dsb_schema
docker rm dsb_schema
