#!/usr/bin/env bash

if [[ "$EUID" -ne 0 ]]; then
  echo "Run this as sudo!"
  exit 1
fi

mkdir -p /etc/docker/
cat > /etc/docker/daemon.json <<EOF
{
    "data-root": "/mnt/nvme1n1/docker/"
}
EOF

sudo systemctl restart docker
