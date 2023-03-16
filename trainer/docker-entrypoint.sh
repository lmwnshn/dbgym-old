#!/bin/bash

set -euxo pipefail

echo '/tmp/core.%h.%e.%t' > /proc/sys/kernel/core_pattern
ulimit -c unlimited

sudo chown -R trainer:trainer /trainer
cd /app
gunicorn "trainer.wsgi:app"