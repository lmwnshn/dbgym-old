#!/usr/bin/env bash

if docker volume inspect tpch_sf10; then
  echo "Volume 'tpch_sf10' exists."
  exit
fi

echo "Volume 'tpch_sf10' doesn't exist, creating."

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
./dbgen -vf -s 10
rm -rf ./tbl_data
mkdir ./tbl_data
mv ./*.tbl ./tbl_data
cd "${ROOT_DIR}"

docker run --volume=tpch_sf10:/tpch_sf10 --name tpch_sf10 busybox true
docker cp ./build/tpch-kit/dbgen/tbl_data/. tpch_sf10:/tpch_sf10
docker rm tpch_sf10
