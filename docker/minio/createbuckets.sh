#!/bin/sh
set -e

# MinIO bucket creation script
# For development: uses default credentials
# For production: set MINIO_ROOT_USER and MINIO_ROOT_PASSWORD environment variables

MC_ALIAS="local"
ENDPOINT="${MINIO_ENDPOINT:-http://minio:9000}"
# Use environment variables with defaults for development only
ACCESS="${MINIO_ROOT_USER:-minioadmin}"
SECRET="${MINIO_ROOT_PASSWORD:-minioadmin}"

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