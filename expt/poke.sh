#!/usr/bin/env bash

sudo --validate
while sleep 300; do sudo --validate; kill -0 "$$" || exit; done 2>/dev/null &

set -euxo pipefail

IS_DEV_MACHINE=0
if [[ "$(hostname --all-fqdns)" == *"db.pdl.local.cmu.edu"* ]]; then
  if ! docker info | grep "Docker Root Dir: /mnt/nvme1n1"; then
    echo "Please set up the nvme drive on the dev machines: 'sudo ./setup/docker/pdl.sh'"
    exit 1
  fi
  IS_DEV_MACHINE=1
fi

if [[ ${IS_DEV_MACHINE} -eq 1 ]]; then
  export DOCKER_DATA_ROOT="/mnt/nvme1n1/docker"
else
  export DOCKER_DATA_ROOT="/var/lib/docker"
fi
export HOSTNAME=$(hostname)

docker compose down --remove-orphans
docker compose build

#docker compose up --detach --wait
#curl -X POST http://0.0.0.0:15721/postgres/clone/
#curl -X POST http://0.0.0.0:15721/postgres/configure/
#curl -X POST http://0.0.0.0:15721/postgres/make/
#curl -X POST http://0.0.0.0:15721/postgres/initdb/
#docker compose down --remove-orphans

docker compose up
