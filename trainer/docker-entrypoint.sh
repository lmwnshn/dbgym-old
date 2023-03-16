#!/bin/bash

set -euxo pipefail

ulimit -c unlimited

sudo chown -R trainer:trainer /trainer
cd /app
gunicorn "trainer.wsgi:app"