# Vulcan LLM NGINX Origin Server - Complete Documentation

## Overview

This is a production-ready NGINX origin server configuration optimized for serving Vulcan LLM pack files, Bloom filters, manifests, and proofs from MinIO object storage.

**Version**: 4.6.0  
**Last Updated**: 2025-11-14  
**NGINX Version Required**: 1.24+

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Features](#key-features)
3. [Configuration Structure](#configuration-structure)
4. [Endpoint Documentation](#endpoint-documentation)
5. [Caching Strategy](#caching-strategy)
6. [Security Features](#security-features)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Deployment Guide](#deployment-guide)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance](#maintenance)

---

## Architecture Overview

### System Design

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Clients   │─────▶│  NGINX Origin    │─────▶│  MinIO Cluster  │
│             │      │  (Port 80/443)   │      │  (4 nodes)      │
└─────────────┘      └──────────────────┘      └─────────────────┘
                              │
                              ├─────▶ Hot Cache (10GB)
                              ├─────▶ Warm Cache (50GB)
                              ├─────▶ Cold Cache (100GB)
                              ├─────▶ Bloom Cache (1GB)
                              └─────▶ Manifest Cache (20GB)
```

### Components

1. **NGINX Origin Server**: High-performance reverse proxy and cache
2. **MinIO Backend**: S3-compatible object storage cluster
3. **Cache Layers**: Multi-tier caching strategy
4. **Monitoring**: Prometheus metrics and logging

---

## Key Features

### ✅ High Availability
- **Multiple upstream backends** with automatic failover
- **Health checking** with configurable fail_timeout
- **Backup cluster** for disaster recovery
- **Read-only replicas** for GET request optimization
- **Connection pooling** with keepalive

### ✅ Performance Optimization
- **Multi-tier caching**: Hot (24h), Warm (7d), Cold (30d)
- **HTTP/2 support** for multiplexing
- **Compression**: Gzip for JSON/text content
- **Range request support** for partial content
- **TCP optimizations**: tcp_nodelay, tcp_nopush, sendfile
- **Cache locking** to prevent thundering herd
- **Background cache updates** for popular content

### ✅ Security
- **Rate limiting** by IP, user agent, and endpoint
- **Connection limits** per IP address
- **Security headers**: CSP, HSTS, X-Frame-Options, etc.
- **Malicious agent blocking**: Scanners, bots, crawlers
- **Request validation**: Method, content-type, URL patterns
- **TLS 1.2/1.3** with modern cipher suites
- **CORS configuration** with proper headers

### ✅ Observability
- **Detailed logging**: Request metrics, performance, security
- **Health check endpoints**: /healthz, /readyz, /livez
- **Metrics endpoint**: Prometheus-compatible
- **Request tracing**: Unique request IDs
- **Cache analytics**: Hit rates, miss rates, performance

### ✅ Scalability
- **Load balancing**: Least connections, consistent hashing
- **Horizontal scaling**: Add more upstream servers
- **Cache scaling**: Configurable cache sizes
- **Connection pooling**: Reuse backend connections
- **Async operations**: Background cache updates

---

## Configuration Structure

### File Organization

```
origin.conf
├── Upstream Configuration (Lines 10-75)
│   ├── minio_backend (primary cluster)
│   ├── minio_backup (disaster recovery)
│   └── minio_readonly (read replicas)
│
├── Cache Configuration (Lines 80-125)
│   ├── hot_cache (10GB, 24h)
│   ├── warm_cache (50GB, 7d)
│   ├── cold_cache (100GB, 30d)
│   ├── bloom_cache (1GB, 1h)
│   └── manifest_cache (20GB, 6h)
│
├── Rate Limiting (Lines 130-155)
│   ├── by_ip (100 req/s)
│   ├── by_user_agent (50 req/s)
│   ├── api_limit (1000 req/s)
│   └── Connection limits
│
├── Security Configuration (Lines 165-220)
│   ├── Geo-IP mapping
│   ├── User agent blocking
│   ├── Request validation
│   └── Content type validation
│
├── Logging Configuration (Lines 225-275)
│   ├── detailed (comprehensive metrics)
│   ├── performance (timing data)
│   ├── security (audit trail)
│   └── cache_analytics (cache stats)
│
├── SSL/TLS Configuration (Lines 280-300)
│
└── Server Blocks (Lines 305+)
    ├── HTTP Server (Port 80)
    └── HTTPS Server (Port 443)
```

---

## Endpoint Documentation

### Core Endpoints

#### 1. `/v1/memory/{hash}` - Pack Files

**Purpose**: Serve embedding pack files (.gpk.zst)

**Method**: GET, HEAD, OPTIONS

**Request**:
```bash
GET /v1/memory/a1b2c3d4e5f6...
Range: bytes=0-1023
If-None-Match: "abc123"
X-Bloom: present
```

**Response**:
```
HTTP/1.1 200 OK
Content-Type: application/octet-stream
Content-Length: 8192
Accept-Ranges: bytes
ETag: "abc123"
X-Cache-Status: HIT
X-Pack-Merkle: 0x...
X-Request-ID: 1a2b3c4d...
```

**Features**:
- ✅ Prefix-based routing (first 2 chars of hash)
- ✅ Range request support
- ✅ ETag validation
- ✅ Hot cache (7-day TTL)
- ✅ Automatic failover to backup
- ✅ Bloom filter integration

**Cache Strategy**:
- Primary: hot_cache (10GB, 7d TTL)
- Key: `$scheme$host$uri$is_args$args$http_range`
- Stale-while-revalidate enabled
- Background updates for popular content

**Performance**:
- Target latency: <50ms (cache hit)
- Target throughput: 10,000+ req/s
- Cache hit rate target: >85%

#### 2. `/v1/bloom/{hash}` - Bloom Filters

**Purpose**: Serve Bloom filter files for fast negative lookups

**Method**: GET, HEAD, OPTIONS

**Features**:
- ✅ Small files (~128 bytes)
- ✅ Aggressive caching (1h TTL)
- ✅ High request rate (1000 burst)
- ✅ Optimized buffering

**Cache Strategy**:
- Primary: bloom_cache (1GB, 1h TTL)
- Minimal buffering (32k buffers)
- High hit rate target: >95%

#### 3. `/v1/manifest/{hash}` - Manifests

**Purpose**: Serve pack manifest files (JSON)

**Method**: GET, HEAD, OPTIONS

**Features**:
- ✅ ETag revalidation
- ✅ Gzip compression
- ✅ 304 Not Modified support
- ✅ 6-hour TTL

**Cache Strategy**:
- Primary: manifest_cache (20GB, 6h TTL)
- Cache key includes ETag
- JSON compression enabled

#### 4. `/proofs/{path}` - Proof Files

**Purpose**: Serve cryptographic proof files

**Method**: GET, HEAD, OPTIONS

**Features**:
- ✅ Long-term caching (30d TTL)
- ✅ Streaming (no buffering)
- ✅ Archive storage tier

**Cache Strategy**:
- Primary: cold_cache (100GB, 30d TTL)
- Stream-based delivery

### Health & Monitoring Endpoints

#### `/healthz` - Health Check

**Purpose**: Basic health check for load balancers

**Response**:
```
HTTP/1.1 200 OK
Content-Type: text/plain
X-Health-Check: pass

ok
```

**Rate Limit**: 10 req/s (burst 20)

#### `/healthz/detailed` - Detailed Health

**Purpose**: Comprehensive health status with JSON

**Response**:
```json
{
  "status": "healthy",
  "version": "4.6.0",
  "timestamp": "1700000000.123",
  "upstream": "available"
}
```

#### `/readyz` - Readiness Probe

**Purpose**: Kubernetes readiness probe

**Response**: 200 OK with "ready"

#### `/livez` - Liveness Probe

**Purpose**: Kubernetes liveness probe

**Response**: 200 OK with "alive"

#### `/metrics` - Prometheus Metrics

**Purpose**: Export metrics for Prometheus

**Access**: Internal networks only (10.0.0.0/8, etc.)

**Metrics**:
- Active connections
- Request rate
- Response times
- Cache hit rates

#### `/nginx_status` - NGINX Status

**Purpose**: NGINX stub_status module

**Access**: Internal networks only

**Response**:
```
Active connections: 291
server accepts handled requests
 16630948 16630948 31070465
Reading: 6 Writing: 179 Waiting: 106
```

### Admin Endpoints

#### `/cache/purge` - Cache Purge

**Purpose**: Purge cache entries

**Method**: POST

**Access**: Internal networks only

**Request**:
```bash
POST /cache/purge
Content-Type: application/json

{
  "pattern": "/v1/memory/a1b2c3*"
}
```

#### `/cache/stats` - Cache Statistics

**Purpose**: Get cache statistics

**Access**: Internal networks only

**Response**:
```json
{
  "hot_cache": "hot_cache",
  "warm_cache": "warm_cache",
  "cold_cache": "cold_cache",
  "bloom_cache": "bloom_cache",
  "manifest_cache": "manifest_cache"
}
```

---

## Caching Strategy

### Multi-Tier Cache Architecture

#### Tier 1: Hot Cache (Memory/SSD)
- **Size**: 10GB
- **TTL**: 24 hours
- **Target**: Frequently accessed pack files
- **Location**: `/var/cache/nginx/hot`
- **Levels**: 1:2
- **Keys Zone**: 512MB
- **Expected Hit Rate**: >85%

#### Tier 2: Warm Cache (SSD)
- **Size**: 50GB
- **TTL**: 7 days
- **Target**: Recently accessed content
- **Location**: `/var/cache/nginx/warm`
- **Expected Hit Rate**: >70%

#### Tier 3: Cold Cache (HDD)
- **Size**: 100GB
- **TTL**: 30 days
- **Target**: Archive content, proofs
- **Location**: `/var/cache/nginx/cold`
- **Expected Hit Rate**: >50%

#### Tier 4: Bloom Cache (Memory)
- **Size**: 1GB
- **TTL**: 1 hour
- **Target**: Bloom filters
- **Location**: `/var/cache/nginx/bloom`
- **Expected Hit Rate**: >95%

#### Tier 5: Manifest Cache (SSD)
- **Size**: 20GB
- **TTL**: 6 hours
- **Target**: Pack manifests
- **Location**: `/var/cache/nginx/manifest`
- **Expected Hit Rate**: >90%

### Cache Key Design

```nginx
# Memory endpoint
proxy_cache_key "$scheme$host$uri$is_args$args$http_range"

# Bloom endpoint
proxy_cache_key "$scheme$host$uri"

# Manifest endpoint
proxy_cache_key "$scheme$host$uri$http_if_none_match"
```

### Cache Optimization Features

1. **Cache Locking**: Prevent duplicate upstream requests
2. **Background Updates**: Refresh popular content asynchronously
3. **Stale-While-Revalidate**: Serve stale content during updates
4. **Cache Bypass**: Bypass cache for specific conditions
5. **Revalidation**: Use ETag for efficient cache validation

### Cache Performance Metrics

**Target Metrics**:
- Overall cache hit rate: >80%
- Hot cache hit rate: >85%
- Bloom cache hit rate: >95%
- Average cache lookup time: <1ms
- Cache miss latency: <50ms

---

## Security Features

### 1. Rate Limiting

```nginx
# By IP address
limit_req_zone $binary_remote_addr zone=by_ip:10m rate=100r/s;

# API endpoint
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=1000r/s;

# Connection limit
limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
```

**Configuration**:
- Default rate: 100 req/s per IP
- API rate: 1000 req/s per IP
- Burst allowance: 500 requests
- Connection limit: 50 concurrent per IP

### 2. Security Headers

```nginx
# CORS
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, HEAD, OPTIONS
Access-Control-Expose-Headers: ETag, X-Pack-Merkle, ...

# Security
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000
Content-Security-Policy: default-src 'none'
```

### 3. Request Validation

**Blocked User Agents**:
- Scanners (nikto, sqlmap, nmap, masscan)
- Malicious tools (havij, acunetix)

**Blocked Patterns**:
- Path traversal: `..`
- Null bytes: `\x00`, `%00`
- SQL injection: `union select`, `drop table`
- XSS: `<script>`

### 4. TLS Configuration

```nginx
# Protocols
TLSv1.2 TLSv1.3

# Ciphers
ECDHE-ECDSA-AES128-GCM-SHA256
ECDHE-RSA-AES128-GCM-SHA256
ECDHE-ECDSA-AES256-GCM-SHA384
ECDHE-RSA-AES256-GCM-SHA384
ECDHE-ECDSA-CHACHA20-POLY1305
ECDHE-RSA-CHACHA20-POLY1305

# Features
- Session caching (50MB)
- OCSP stapling
- Perfect forward secrecy
```

### 5. Access Control

**Admin Endpoints**:
- Restricted to 127.0.0.1 only
- Basic authentication required

**Metrics Endpoints**:
- Restricted to internal networks
- IP whitelist: 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16

---

## Performance Optimization

### 1. Connection Management

```nginx
# Keepalive
keepalive_timeout 65s;
keepalive_requests 1000;

# Upstream keepalive
keepalive 256;
keepalive_requests 10000;
keepalive_timeout 300s;

# HTTP version
proxy_http_version 1.1;
```

### 2. TCP Optimizations

```nginx
tcp_nodelay on;      # Disable Nagle's algorithm
tcp_nopush on;       # Send headers in one packet
sendfile on;         # Zero-copy file transfer
sendfile_max_chunk 512k;
```

### 3. Buffer Tuning

```nginx
# Client buffers
client_body_buffer_size 128k;
client_header_buffer_size 1k;
large_client_header_buffers 4 16k;

# Proxy buffers (memory endpoint)
proxy_buffer_size 128k;
proxy_buffers 8 128k;
proxy_busy_buffers_size 256k;
```

### 4. Worker Configuration

Add to `nginx.conf`:
```nginx
worker_processes auto;
worker_rlimit_nofile 65535;
worker_priority -5;

events {
    worker_connections 10000;
    use epoll;
    multi_accept on;
}
```

### 5. File Descriptor Limits

```bash
# /etc/security/limits.conf
nginx soft nofile 65535
nginx hard nofile 65535
```

---

## Monitoring and Observability

### Log Files

```
/var/log/nginx/
├── origin-access.log          # Detailed access log
├── origin-error.log           # Error log
├── origin-security.log        # Security events
├── origin-performance.log     # Performance metrics
├── origin-https-access.log    # HTTPS access log
├── origin-https-error.log     # HTTPS errors
├── healthcheck.log            # Health check logs
└── unmatched.log             # Unmatched requests
```

### Log Format Fields

**Detailed Log Format**:
- `$remote_addr` - Client IP
- `$request` - Full request line
- `$status` - Response status
- `$request_time` - Total request time
- `$upstream_response_time` - Backend response time
- `$upstream_cache_status` - Cache status (HIT/MISS/BYPASS)
- `$request_id` - Unique request ID
- `$bytes_sent` - Response size

### Metrics Collection

**Prometheus Metrics** (via nginx-prometheus-exporter):
- `nginx_http_requests_total` - Total requests
- `nginx_http_request_duration_seconds` - Request latency
- `nginx_http_upstream_response_time_seconds` - Upstream latency
- `nginx_cache_hit_rate` - Cache hit rate
- `nginx_connections_active` - Active connections
- `nginx_connections_reading` - Reading connections
- `nginx_connections_writing` - Writing connections
- `nginx_connections_waiting` - Idle connections

**Custom Metrics**:
```bash
# Cache hit rate
awk '$11 ~ /HIT/ {hit++} $11 ~ /MISS/ {miss++} END {print hit/(hit+miss)*100"%"}' \
  /var/log/nginx/origin-access.log

# Average response time
awk '{sum+=$15; count++} END {print sum/count"s"}' \
  /var/log/nginx/origin-access.log

# Top requested URLs
awk '{print $7}' /var/log/nginx/origin-access.log | \
  sort | uniq -c | sort -rn | head -20
```

### Grafana Dashboards

**Dashboard 1: Overview**
- Request rate (req/s)
- Error rate (%)
- Average response time (ms)
- Cache hit rate (%)
- Active connections
- Bandwidth (MB/s)

**Dashboard 2: Cache Performance**
- Hit rate by cache tier
- Miss rate by endpoint
- Cache size utilization
- Eviction rate
- Top cached URLs

**Dashboard 3: Backend Health**
- Upstream response time
- Backend availability
- Failover events
- Connection pool usage

**Dashboard 4: Security**
- Blocked requests
- Rate limit hits
- Suspicious patterns
- Geographic distribution

---

## Deployment Guide

### Prerequisites

```bash
# NGINX 1.24+ with required modules
nginx -V 2>&1 | grep -o 'with-[^ ]*'

Required modules:
- http_ssl_module
- http_v2_module
- http_realip_module
- http_stub_status_module
- http_gzip_static_module

# System requirements
- CPU: 8+ cores
- RAM: 16GB+ (for cache)
- Disk: 200GB+ SSD (for cache)
- Network: 10Gbps+
```

### Installation Steps

#### 1. Install NGINX

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y nginx nginx-extras

# RHEL/CentOS
sudo yum install -y nginx

# Verify installation
nginx -v
nginx -V
```

#### 2. Create Directory Structure

```bash
# Cache directories
sudo mkdir -p /var/cache/nginx/{hot,warm,cold,bloom,manifest}
sudo chown -R nginx:nginx /var/cache/nginx
sudo chmod -R 755 /var/cache/nginx

# Log directories
sudo mkdir -p /var/log/nginx
sudo chown -R nginx:nginx /var/log/nginx

# SSL certificates
sudo mkdir -p /etc/nginx/ssl
sudo chmod 700 /etc/nginx/ssl

# Static files
sudo mkdir -p /var/www/html
sudo mkdir -p /var/www/static
sudo chown -R nginx:nginx /var/www
```

#### 3. Deploy Configuration

```bash
# Backup existing config
sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup

# Copy origin.conf
sudo cp origin.conf /etc/nginx/sites-available/origin.conf

# Create symlink
sudo ln -s /etc/nginx/sites-available/origin.conf \
           /etc/nginx/sites-enabled/origin.conf

# Remove default site
sudo rm -f /etc/nginx/sites-enabled/default
```

#### 4. Update nginx.conf

```bash
sudo nano /etc/nginx/nginx.conf
```

Add/modify:
```nginx
user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
pid /run/nginx.pid;

events {
    worker_connections 10000;
    use epoll;
    multi_accept on;
}

http {
    # Basic settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;
    
    # MIME types
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log warn;
    
    # Gzip
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript
               application/json application/javascript application/xml+rss;
    
    # Include site configs
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

#### 5. SSL Certificates

```bash
# Self-signed (development)
sudo openssl req -x509 -nodes -days 365 \
  -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/key.pem \
  -out /etc/nginx/ssl/cert.pem

# Let's Encrypt (production)
sudo certbot --nginx -d origin.vulcanami.io
```

#### 6. Validate Configuration

```bash
# Test configuration
sudo nginx -t

# Expected output:
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful
```

#### 7. Start NGINX

```bash
# Start NGINX
sudo systemctl start nginx

# Enable on boot
sudo systemctl enable nginx

# Check status
sudo systemctl status nginx

# Verify it's listening
sudo netstat -tlnp | grep nginx
# Should show :80 and :443
```

#### 8. Verify Deployment

```bash
# Health check
curl http://localhost/healthz
# Expected: ok

# Detailed health
curl http://localhost/healthz/detailed
# Expected: JSON with status

# Metrics
curl http://localhost/metrics
# Expected: Prometheus metrics

# Test memory endpoint (replace hash)
curl -I http://localhost/v1/memory/a1b2c3...
```

### Post-Deployment Tasks

#### 1. Configure Log Rotation

```bash
sudo nano /etc/logrotate.d/nginx
```

```
/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 nginx adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

#### 2. Setup Monitoring

```bash
# Install nginx-prometheus-exporter
wget https://github.com/nginxinc/nginx-prometheus-exporter/releases/download/v0.11.0/nginx-prometheus-exporter_0.11.0_linux_amd64.tar.gz
tar xzf nginx-prometheus-exporter_0.11.0_linux_amd64.tar.gz
sudo mv nginx-prometheus-exporter /usr/local/bin/

# Create systemd service
sudo nano /etc/systemd/system/nginx-exporter.service
```

```ini
[Unit]
Description=NGINX Prometheus Exporter
After=network.target

[Service]
Type=simple
User=nginx
ExecStart=/usr/local/bin/nginx-prometheus-exporter \
  -nginx.scrape-uri=http://localhost/nginx_status
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Start exporter
sudo systemctl daemon-reload
sudo systemctl start nginx-exporter
sudo systemctl enable nginx-exporter
```

#### 3. Configure Firewall

```bash
# UFW
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# iptables
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

#### 4. Performance Tuning

```bash
# Increase file descriptor limits
echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf

# TCP tuning
sudo nano /etc/sysctl.conf
```

Add:
```
net.ipv4.tcp_fin_timeout = 15
net.ipv4.tcp_tw_reuse = 1
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
```

Apply:
```bash
sudo sysctl -p
```

---

## Troubleshooting

### Common Issues

#### Issue 1: High Cache Miss Rate

**Symptoms**: Cache hit rate <70%

**Diagnosis**:
```bash
# Check cache stats
grep "cache" /var/log/nginx/origin-access.log | \
  awk '{print $NF}' | sort | uniq -c

# Check cache size
du -sh /var/cache/nginx/*
```

**Solutions**:
1. Increase cache size
2. Increase TTL values
3. Enable cache warming
4. Check cache key configuration

#### Issue 2: High Response Time

**Symptoms**: Response time >100ms

**Diagnosis**:
```bash
# Average response time
awk '{sum+=$15; count++} END {print sum/count}' \
  /var/log/nginx/origin-access.log

# Slow requests
awk '$15 > 1' /var/log/nginx/origin-access.log
```

**Solutions**:
1. Check upstream response time
2. Verify cache is working
3. Increase worker processes
4. Check network latency
5. Review buffer sizes

#### Issue 3: 502/503 Errors

**Symptoms**: Frequent 502/503 responses

**Diagnosis**:
```bash
# Check error log
sudo tail -f /var/log/nginx/origin-error.log

# Check upstream health
curl http://localhost/nginx_status

# Test backend
curl -I http://minio:9000/bucket/key
```

**Solutions**:
1. Verify MinIO is running
2. Check upstream configuration
3. Increase timeouts
4. Add more backend servers
5. Review error logs

#### Issue 4: Rate Limit Errors (429)

**Symptoms**: Clients receiving 429 responses

**Diagnosis**:
```bash
# Count 429 responses
grep " 429 " /var/log/nginx/origin-access.log | wc -l

# Check which IPs are hitting limits
grep " 429 " /var/log/nginx/origin-access.log | \
  awk '{print $1}' | sort | uniq -c | sort -rn
```

**Solutions**:
1. Increase rate limit values
2. Whitelist legitimate clients
3. Implement token bucket
4. Add more endpoints
5. Use CDN for static content

#### Issue 5: Memory Usage High

**Symptoms**: High memory consumption

**Diagnosis**:
```bash
# Check process memory
ps aux | grep nginx

# Check cache size
du -sh /var/cache/nginx

# Check shared zones
nginx -T | grep zone
```

**Solutions**:
1. Reduce cache sizes
2. Lower TTL values
3. Enable cache eviction
4. Increase worker memory
5. Review buffer sizes

### Debug Commands

```bash
# Reload configuration
sudo nginx -s reload

# Test configuration
sudo nginx -t

# View configuration
sudo nginx -T

# Check error log
sudo tail -f /var/log/nginx/origin-error.log

# Check access log
sudo tail -f /var/log/nginx/origin-access.log

# Monitor connections
watch -n 1 'curl -s http://localhost/nginx_status'

# Check cache statistics
find /var/cache/nginx -type f | wc -l

# Monitor cache usage
watch -n 5 'du -sh /var/cache/nginx/*'

# Test endpoint
curl -v http://localhost/v1/memory/hash

# Check upstream
curl -v http://minio:9000/bucket/key

# Trace request
curl -H "X-Request-ID: test-123" http://localhost/v1/memory/hash
grep "test-123" /var/log/nginx/origin-access.log
```

---

## Maintenance

### Daily Tasks

```bash
# Check error log
sudo tail -100 /var/log/nginx/origin-error.log

# Check disk space
df -h /var/cache/nginx

# Monitor performance
curl http://localhost/nginx_status

# Check for 5xx errors
grep " 5[0-9][0-9] " /var/log/nginx/origin-access.log | tail -20
```

### Weekly Tasks

```bash
# Analyze cache hit rate
awk '$11 ~ /HIT/ {hit++} $11 ~ /MISS/ {miss++} END {
  print "Hit rate:", hit/(hit+miss)*100"%"
}' /var/log/nginx/origin-access.log.1

# Review top URLs
awk '{print $7}' /var/log/nginx/origin-access.log.1 | \
  sort | uniq -c | sort -rn | head -20

# Check slow requests
awk '$15 > 1 {print $0}' /var/log/nginx/origin-access.log.1 | \
  wc -l

# Review security log
sudo tail -100 /var/log/nginx/origin-security.log
```

### Monthly Tasks

```bash
# Update NGINX
sudo apt update && sudo apt upgrade nginx

# Review configuration
sudo nginx -T > /tmp/nginx-config-$(date +%Y%m%d).txt

# Analyze traffic patterns
# (Use log analysis tools)

# Review SSL certificates
sudo openssl x509 -in /etc/nginx/ssl/cert.pem -noout -dates

# Capacity planning
# Review metrics and plan scaling
```

### Backup and Restore

#### Backup

```bash
# Backup configuration
sudo tar czf nginx-config-backup-$(date +%Y%m%d).tar.gz \
  /etc/nginx/ \
  /var/log/nginx/ \
  --exclude=/var/log/nginx/*.log

# Backup SSL certificates
sudo tar czf nginx-ssl-backup-$(date +%Y%m%d).tar.gz \
  /etc/nginx/ssl/
```

#### Restore

```bash
# Restore configuration
sudo tar xzf nginx-config-backup-20251114.tar.gz -C /

# Test configuration
sudo nginx -t

# Reload
sudo nginx -s reload
```

---

## Performance Benchmarks

### Target Metrics

```
Throughput:
- GET requests: 10,000+ req/s (cache hit)
- GET requests: 1,000+ req/s (cache miss)

Latency:
- P50: <20ms (cache hit), <50ms (cache miss)
- P95: <50ms (cache hit), <100ms (cache miss)
- P99: <100ms (cache hit), <200ms (cache miss)

Cache Performance:
- Hit rate: >80% overall
- Hot cache: >85%
- Bloom cache: >95%

Resource Usage:
- CPU: <50% average
- Memory: <8GB
- Disk I/O: <100MB/s
```

### Benchmark Tools

```bash
# Apache Bench
ab -n 10000 -c 100 http://localhost/v1/memory/hash

# wrk
wrk -t 12 -c 400 -d 30s http://localhost/v1/memory/hash

# hey
hey -n 10000 -c 100 http://localhost/v1/memory/hash

# Custom script
for i in {1..1000}; do
  curl -s -w "%{http_code} %{time_total}s\n" \
    http://localhost/v1/memory/hash > /dev/null
done
```

---

## Support and Resources

### Internal Support
- **Email**: nginx-support@vulcanami.io
- **Slack**: #nginx-support
- **Documentation**: https://docs.vulcanami.io/nginx

### External Resources
- **NGINX Docs**: https://nginx.org/en/docs/
- **NGINX Performance**: https://www.nginx.com/blog/tuning-nginx/
- **NGINX Security**: https://www.nginx.com/blog/nginx-security-best-practices/

---

## Change Log

### Version 4.6.0 (2025-11-14)
- ✨ Complete configuration overhaul
- ✨ Multi-tier caching strategy
- ✨ Advanced security features
- ✨ Comprehensive monitoring
- ✨ Load balancing with failover
- ✨ Rate limiting and DDoS protection
- ✨ Performance optimization
- ✨ Detailed documentation

---

**For questions or issues, contact: nginx-support@vulcanami.io**