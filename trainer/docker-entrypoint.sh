#!/bin/bash

set -euxo pipefail

ulimit -c unlimited

sudo chown -R trainer:trainer /trainer
sudo chown -R trainer:trainer /trainer_db
cd /app
gunicorn "trainer.wsgi:app"