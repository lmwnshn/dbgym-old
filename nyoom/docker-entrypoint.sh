#!/bin/bash

set -euxo pipefail

sudo chown -R nyoom:nyoom /nyoom
cp -r /nyoom_default/* /nyoom

cd /app
gunicorn "nyoom_flask.wsgi:app"