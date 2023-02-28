#!/usr/bin/env bash

sudo --validate
while sleep 300; do sudo --validate; kill -0 "$$" || exit; done 2>/dev/null &

set -euxo pipefail

IS_DEV_MACHINE=0
if [[ "$(hostname --all-fqdns)" == *"db.pdl.local.cmu.edu"* ]]; then
  if ! docker info | grep "Docker Root Dir: /mnt/nvme0n1"; then
    echo "Please set up the nvme drive on the dev machines: 'sudo ./setup/docker/pdl.sh'"
    exit 1
  fi
  IS_DEV_MACHINE=1
fi

if [[ ${IS_DEV_MACHINE} -eq 1 ]]; then
  export DOCKER_DATA_ROOT="/mnt/nvme0n1/docker"
else
  export DOCKER_DATA_ROOT="/var/lib/docker"
fi
export HOSTNAME=$(hostname)

sudo apt install make gcc
./setup/tpch/tpch_sf1.sh
./setup/tpch/tpch_sf10.sh
./setup/tpch/tpch_schema.sh
./setup/tpch/tpch_queries.sh

docker compose down --remove-orphans
docker compose build

docker compose --profile gym build
docker compose --profile gym up

docker compose down --remove-orphans

# TODO(WAN): This is AWFUL. But bind mounts are even worse.
rm -rf ./artifact/dbgym
mkdir -p ./artifact/dbgym
sudo cp -r "${DOCKER_DATA_ROOT}"/volumes/dbgym_dbgym/_data/artifact ./artifact/dbgym
sudo chown -R "${USER}":"${USER}" ./artifact
