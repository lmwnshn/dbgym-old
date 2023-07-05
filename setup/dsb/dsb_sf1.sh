#!/usr/bin/env bash

if docker volume inspect dsb_sf1; then
  echo "Volume 'dsb_sf1' exists."
  exit
fi

echo "Volume 'dsb_sf1' doesn't exist, creating."

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

cd ./build/dsb/code/tools
rm -rf ./tbl_data
mkdir ./tbl_data
./dsdgen -dir tbl_data -terminate n -scale 1
cd "${ROOT_DIR}"

docker run --volume=dsb_sf1:/dsb_sf1 --name dsb_sf1 busybox true
docker cp --quiet ./build/dsb/code/tools/tbl_data/. dsb_sf1:/dsb_sf1
docker rm dsb_sf1
