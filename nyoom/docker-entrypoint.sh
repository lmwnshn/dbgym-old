#!/bin/bash

set -euxo pipefail

sudo chown -R nyoom:nyoom /nyoom
cp -r /nyoom_default/* /nyoom
cd /nyoom
python3 -u -m nyoom