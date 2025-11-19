#!/bin/sh
set -e

MC_ALIAS="local"
ENDPOINT="http://minio:9000"
ACCESS="minioadmin"
SECRET="minioadmin"

mc alias set $MC_ALIAS $ENDPOINT $ACCESS $SECRET

mc mb -p $MC_ALIAS/graphix-vulcan-use1 || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/origin || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/archive || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/hot || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/lake || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/proofs || true
mc mb -p $MC_ALIAS/graphix-vulcan-use1/config || true

mc version enable $MC_ALIAS/graphix-vulcan-use1

echo "MinIO buckets ready."