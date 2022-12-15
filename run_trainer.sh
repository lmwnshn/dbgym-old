#!/usr/bin/env bash

python3 -m flask --app dbgym.trainer.trainer --debug run --no-reload --host=0.0.0.0
