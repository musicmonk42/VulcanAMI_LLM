#!/bin/bash
################################################################################
# Secret Generation Script
# 
# Generates secure random secrets for VulcanAMI deployment
# Output can be appended to .env file
################################################################################

set -e

echo "# VulcanAMI Generated Secrets"
echo "# Generated on: $(date)"
echo "# SECURITY: Keep this file secure and never commit to git!"
echo ""

echo "# Core Application Secrets"
echo "JWT_SECRET_KEY=$(openssl rand -base64 48 | tr -d '\n' | tr -d '+/')"
echo "BOOTSTRAP_KEY=$(openssl rand -base64 32 | tr -d '\n' | tr -d '+/')"
echo ""

echo "# Database Secrets"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d '\n' | tr -d '+/')"
echo ""

echo "# Redis Secrets"
echo "REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d '\n' | tr -d '+/')"
echo ""

echo "# MinIO Object Storage Secrets"
echo "MINIO_ROOT_USER=admin"
echo "MINIO_ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d '\n' | tr -d '+/')"
echo ""

echo "# Grafana Secrets"
echo "GRAFANA_USER=admin"
echo "GRAFANA_PASSWORD=$(openssl rand -base64 24 | tr -d '\n' | tr -d '+/')"
echo ""

echo "# Configuration"
echo "ENVIRONMENT=development"
echo "LOG_LEVEL=INFO"
echo "VERSION=latest"
echo ""

echo "# Docker/Container Network Binding"
echo "# Use 0.0.0.0 for containers (required for Docker networking)"
echo "# Use 127.0.0.1 for bare-metal development (more secure)"
echo "HOST=0.0.0.0"
echo "API_HOST=0.0.0.0"
echo ""

echo "# Optional: HuggingFace Model Pinning (recommended for production)"
echo "# Find commit hashes at: https://huggingface.co/<model-name>/commits/main"
echo "# VULCAN_TEXT_MODEL_REVISION="
echo "# VULCAN_AUDIO_MODEL_REVISION="
echo "# VULCAN_BERT_MODEL_REVISION="
echo ""
