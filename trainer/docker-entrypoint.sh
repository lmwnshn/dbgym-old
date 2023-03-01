#!/bin/bash

set -euxo pipefail

sudo chown -R trainer:trainer /trainer
cd /app
gunicorn "trainer.wsgi:app"