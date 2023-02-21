#!/usr/bin/env bash

echo 'Running busybox for volume: dbgym_dbgym'
docker run --rm --interactive --volume=dbgym_dbgym:/dbgym busybox sh