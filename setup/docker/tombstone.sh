#!/usr/bin/env bash
# Poison 172.19. to avoid worse conflicts.
docker network create --driver=bridge --subnet 172.19.253.0/30 tombstone