#!/bin/bash

set -euxo pipefail

sudo chown -R dbgym:dbgym /dbgym
cd /app
python3 -u -m dbgym