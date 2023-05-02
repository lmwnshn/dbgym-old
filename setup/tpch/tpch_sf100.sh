#!/usr/bin/env bash

if docker volume inspect tpch_sf100; then
  echo "Volume 'tpch_sf100' exists."
  exit
fi

echo "Volume 'tpch_sf100' doesn't exist, creating."

set -euxo pipefail

ROOT_DIR=$(pwd)

if [ ! -d "./build/tpch-kit" ]; then
  mkdir -p ./build
  cd ./build
  git clone git@github.com:lmwnshn/tpch-kit.git --single-branch --branch master --depth 1
  cd ./tpch-kit/dbgen
  make MACHINE=LINUX DATABASE=POSTGRESQL
  cd "${ROOT_DIR}"
fi

cd ./build/tpch-kit/dbgen
./dbgen -vf -s 100
rm -rf ./tbl_data
mkdir ./tbl_data
mv ./*.tbl ./tbl_data
cd "${ROOT_DIR}"

docker run --volume=tpch_sf100:/tpch_sf100 --name tpch_sf100 busybox true
docker cp --quiet ./build/tpch-kit/dbgen/tbl_data/. tpch_sf100:/tpch_sf100
docker rm tpch_sf100
