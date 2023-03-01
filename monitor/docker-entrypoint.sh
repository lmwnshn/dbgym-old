#!/bin/bash

set -euxo pipefail
cd /app
gunicorn "monitor.wsgi:app"