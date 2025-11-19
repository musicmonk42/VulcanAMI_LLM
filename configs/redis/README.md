# Vulcan LLM Redis Configuration - Complete Documentation

## Overview

This package contains three fully enhanced, production-ready Redis configuration files for the Vulcan LLM system:

1. **exporter.env** - Redis Exporter Environment Configuration
2. **redis.conf** - Redis Server Configuration
3. **keys_ttl.yml** - TTL and Eviction Policy Configuration

## Version Information

- **Version**: 4.6.0
- **Last Updated**: 2025-11-14
- **Redis Version**: 7.2+
- **Redis Exporter Version**: Latest
- **Environment**: Production-ready

---

## 📁 File 1: exporter.env

### Overview
Comprehensive Redis exporter configuration for Prometheus metrics collection and monitoring.

### Key Features

#### Connection Management
- ✅ Support for single, multiple, and clustered Redis instances
- ✅ Redis Sentinel configuration
- ✅ TLS/SSL encryption support
- ✅ Authentication with username/password or password files
- ✅ Connection pooling and timeout configuration

#### Metrics Collection
- ✅ **Keyspace Monitoring**: Track all key patterns with custom grouping
- ✅ **Command Statistics**: Per-command performance metrics
- ✅ **Slowlog Monitoring**: Automatic slow query detection
- ✅ **Latency Histograms**: Detailed latency distribution tracking
- ✅ **Memory Analytics**: Memory usage sampling and statistics
- ✅ **Client Metrics**: Connection and client list monitoring
- ✅ **Stream Monitoring**: Redis Streams metrics

#### Vulcan-Specific Configuration
```
Key Pattern Groups:
- packs=pack:*               (Embedding packs)
- blooms=pack_bloom:*        (Bloom filters)
- manifests=manifest:*       (Pack manifests)
- pins=pin:*                 (Pinned shards)
- ids=ids:*                  (ID lookups)
- docs=doc:*                 (Document cache)
- negative=neg:*             (Negative cache)
```

#### Advanced Features
- 🔍 Script duration monitoring
- 📊 Configuration metrics export
- 🔐 TLS server support for metrics endpoint
- 🏷️ Constant labels for multi-cluster environments
- 📈 Go runtime and build info metrics
- ⚡ Performance-optimized sampling

### Configuration Sections

1. **Redis Connection Settings** (Lines 10-40)
2. **Exporter Server Settings** (Lines 45-55)
3. **Metric Collection Settings** (Lines 60-90)
4. **Redis INFO Sections** (Lines 95-105)
5. **Script and Lua Monitoring** (Lines 110-125)
6. **Command and Slowlog Monitoring** (Lines 130-145)
7. **Memory and Key Analysis** (Lines 150-160)
8. **Performance and Optimization** (Lines 165-175)
9. **Logging and Debugging** (Lines 195-210)
10. **Security and Access Control** (Lines 215-225)
11. **Prometheus Integration** (Lines 230-240)
12. **Vulcan-Specific Metrics** (Lines 280-295)

### Usage

```bash
# Load environment variables
source exporter.env

# Run Redis exporter
redis_exporter

# Access metrics
curl http://localhost:9121/metrics
```

---

## 📁 File 2: redis.conf

### Overview
Production-hardened Redis server configuration with comprehensive security, performance, and reliability features.

### Key Features

#### Security Hardening
- ✅ **Command Renaming**: Dangerous commands disabled or renamed
  - FLUSHALL → ""
  - FLUSHDB → ""
  - CONFIG → ""
  - KEYS → ""
  - DEBUG → ""
  - SAVE → ""
- ✅ **ACL Support**: Fine-grained access control (Redis 6.0+)
- ✅ **TLS/SSL Encryption**: At-rest and in-transit
- ✅ **Protected Mode**: Configurable network security
- ✅ **Client Output Limits**: Prevent memory exhaustion
- ✅ **Connection Limits**: Maximum client protection

#### Performance Optimization
- ✅ **I/O Threading**: Multi-threaded network I/O (4 threads)
- ✅ **Memory Management**: 
  - 2GB maxmemory limit
  - allkeys-lfu eviction policy
  - Active defragmentation
  - Lazy freeing
- ✅ **Persistence**: Hybrid RDB+AOF
  - RDB: 900s/1 change, 300s/10 changes, 60s/10000 changes
  - AOF: everysec fsync
  - Auto-rewrite optimization
- ✅ **Data Structure Optimization**: 
  - Optimized hash, list, set, zset encodings
  - Memory-efficient representations
  - Compression enabled

#### High Availability
- ✅ **Replication Support**: Master-replica configuration
- ✅ **Sentinel Integration**: Automatic failover
- ✅ **Cluster Mode**: Redis Cluster support
- ✅ **Backlog**: 256MB replication backlog
- ✅ **Minimum Replicas**: Configurable write protection

#### Monitoring and Observability
- ✅ **Slow Log**: 10ms threshold, 128 entries
- ✅ **Latency Monitoring**: 100ms threshold
- ✅ **Keyspace Notifications**: Expiration events (Ex)
- ✅ **INFO Metrics**: Comprehensive server stats

### Memory Configuration

```
maxmemory: 2gb
maxmemory-policy: allkeys-lfu
maxmemory-samples: 5

Active Defragmentation:
- Threshold lower: 10%
- Threshold upper: 100%
- Cycle min: 1%
- Cycle max: 25%
```

### Persistence Configuration

```
RDB Snapshots:
- save 900 1    (15 minutes, 1 change)
- save 300 10   (5 minutes, 10 changes)
- save 60 10000 (1 minute, 10000 changes)

AOF:
- appendonly: yes
- appendfsync: everysec
- aof-use-rdb-preamble: yes
```

### Configuration Sections

1. **Network and Connection** (Lines 10-30)
2. **TLS/SSL Configuration** (Lines 35-70)
3. **General Configuration** (Lines 75-100)
4. **Snapshotting (RDB)** (Lines 105-125)
5. **Replication** (Lines 130-165)
6. **Security** (Lines 170-200)
7. **Memory Management** (Lines 205-235)
8. **AOF Persistence** (Lines 240-265)
9. **Lua Scripting** (Lines 270-280)
10. **Redis Cluster** (Lines 285-310)
11. **Slow Log** (Lines 315-325)
12. **Latency Monitoring** (Lines 330-335)
13. **Event Notification** (Lines 340-355)
14. **Advanced Configuration** (Lines 360-390)
15. **Threaded I/O** (Lines 395-405)

### Usage

```bash
# Start Redis with custom config
redis-server /path/to/redis.conf

# Verify configuration
redis-cli CONFIG GET maxmemory
redis-cli CONFIG GET maxmemory-policy

# Monitor performance
redis-cli --latency
redis-cli --stat
redis-cli SLOWLOG GET 10
```

---

## 📁 File 3: keys_ttl.yml

### Overview
Comprehensive TTL and eviction policy configuration with tier-based lifecycle management.

### Key Features

#### 6-Tier Key Organization

**TIER 0: PINNED KEYS** (No Eviction, No TTL)
- `pack:pack-*` - Core embedding packs (50K keys, 400MB)
- `pin:shard:*` - Pinned shard IDs (2.5K keys, 1MB)
- `config:*` - System configuration (100 keys)
- `model:embeddings:*` - Model metadata (50 keys)

**TIER 1: HOT CACHE** (Short TTL, LFU Eviction)
- `pack_bloom:pack-*` - Bloom filters (1 hour TTL, 50K keys, 6MB)
- `manifest:sha256:*` - Pack manifests (1 hour TTL, 50K keys, 100MB)
- `query:cache:*` - Query results (30 min TTL, 100K keys, 400MB)
- `embedding:cache:*` - Cached embeddings (2 hour TTL, 500K keys, 3GB)

**TIER 2: WARM CACHE** (Medium TTL, LFU Eviction)
- `ids:*:*:sig-*` - ID lookups (60-300s TTL, 1M keys, 500MB)
- `doc:sha256:*` - Document cache (30-120s TTL, 2M keys, 32GB)
- `metadata:doc:*` - Document metadata (10 min TTL)
- `user:session:*` - User sessions (30 min TTL)

**TIER 3: COLD CACHE** (Long TTL or Negative)
- `neg:sha256:*` - Negative cache (3s TTL, 100K keys, 6MB)
- `archive:*` - Archived data (24 hour TTL)
- `stats:daily:*` - Daily statistics (7 day TTL)
- `backup:snapshot:*` - Backup snapshots (30 day TTL)

**TIER 4: TEMPORARY/EPHEMERAL**
- `lock:*` - Distributed locks (30s TTL)
- `task:queue:*` - Task queues (1 hour TTL)
- `rate_limit:*` - Rate limiters (60s TTL)
- `temp:*` - Temporary data (5 min TTL)

**TIER 5: PUBSUB AND STREAMING**
- `stream:events:*` - Event streams (1 hour TTL)
- `pubsub:channel:*` - Pub/Sub channels (no TTL)
- `stream:logs:*` - Log streams (24 hour TTL)

**TIER 6: ANALYTICS AND METRICS**
- `metric:*` - Application metrics (1 hour TTL)
- `counter:*` - Counters (24 hour TTL)
- `histogram:*` - Histogram data (1 hour TTL)

#### Lifecycle Management
- ✅ Automatic TTL extension based on access patterns
- ✅ Key migration between tiers
- ✅ Orphaned keys cleanup
- ✅ Expired keys cleanup
- ✅ Hot-to-warm and warm-to-cold transitions

#### Monitoring and Alerting
```yaml
Tracked Metrics:
- Key count by pattern
- Memory usage by pattern
- Access frequency
- Cache hit rate
- Eviction rate
- TTL distribution
- Key size distribution
- Replication lag

Alerts:
- High eviction rate (>1000/sec)
- Low hit rate (<70%)
- Memory pressure (>90%)
- Missing pinned keys
- High negative cache hit rate (>30%)
- Stuck locks (>60s)
- Key size anomalies
```

#### Optimization Strategies
- 🗜️ **Compression**: LZF algorithm for large values
- 🔄 **Deduplication**: SHA256-based content dedup
- 📊 **Memory Optimization**: Efficient encodings, lazy free
- 📈 **Access Pattern Analysis**: 24-hour analysis cycle

#### Backup and Recovery
```yaml
Schedules:
- Pinned keys: Hourly (168 backups retained)
- Hot cache: Daily (7 backups retained)
- Full backup: Weekly (4 backups retained)

Storage: S3 with encryption

Recovery Priority:
1. pack:pack-*
2. pin:shard:*
3. config:*
4. pack_bloom:*
5. manifest:*
```

#### Security and Compliance
- 🔐 Encryption at rest and in transit
- 🔑 Role-based access control per pattern
- 📝 Audit logging for access and mutations
- 🛡️ GDPR compliance with PII protection

### Configuration Sections

1. **Metadata and Documentation** (Lines 10-25)
2. **Global Defaults** (Lines 30-40)
3. **Policy Recommendations** (Lines 45-75)
4. **Tier 0: Pinned Keys** (Lines 80-145)
5. **Tier 1: Hot Cache** (Lines 150-240)
6. **Tier 2: Warm Cache** (Lines 245-320)
7. **Tier 3: Cold Cache** (Lines 325-380)
8. **Tier 4: Temporary** (Lines 385-425)
9. **Tier 5: Streaming** (Lines 430-465)
10. **Tier 6: Analytics** (Lines 470-500)
11. **Lifecycle Management** (Lines 505-530)
12. **Monitoring and Alerting** (Lines 535-595)
13. **Optimization Strategies** (Lines 600-645)
14. **Backup and Recovery** (Lines 650-690)
15. **Security and Compliance** (Lines 695-725)

### Usage

```bash
# Parse and validate configuration
python validate_ttl_config.py keys_ttl.yml

# Apply TTL policies
python apply_ttl_policies.py keys_ttl.yml

# Monitor key patterns
python monitor_keys.py keys_ttl.yml

# Generate report
python ttl_report.py keys_ttl.yml
```

---

## 🚀 Deployment Guide

### Prerequisites

```bash
# Redis 7.2+ installed
redis-server --version

# Redis exporter installed
redis_exporter --version

# Python 3.8+ (for TTL management scripts)
python --version
```

### Step 1: Deploy Redis Server

```bash
# Backup existing configuration
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup

# Copy new configuration
sudo cp redis.conf /etc/redis/redis.conf

# Validate configuration
redis-server /etc/redis/redis.conf --test-memory 2

# Restart Redis
sudo systemctl restart redis-server

# Verify
redis-cli ping
redis-cli INFO server
```

### Step 2: Deploy Redis Exporter

```bash
# Copy exporter configuration
sudo cp exporter.env /etc/redis/exporter.env

# Create systemd service
sudo cat > /etc/systemd/system/redis_exporter.service << 'EOF'
[Unit]
Description=Redis Exporter
After=network.target redis.service

[Service]
Type=simple
EnvironmentFile=/etc/redis/exporter.env
ExecStart=/usr/local/bin/redis_exporter
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable redis_exporter
sudo systemctl start redis_exporter

# Verify
curl http://localhost:9121/metrics
```

### Step 3: Apply TTL Policies

```bash
# Copy TTL configuration
sudo cp keys_ttl.yml /etc/redis/keys_ttl.yml

# Apply policies (implementation depends on your automation)
# Example using custom script:
python apply_ttl_policies.py /etc/redis/keys_ttl.yml

# Set up monitoring
python setup_ttl_monitoring.py /etc/redis/keys_ttl.yml
```

### Step 4: Configure Monitoring

```bash
# Prometheus configuration
cat >> /etc/prometheus/prometheus.yml << 'EOF'
  - job_name: 'redis_vulcan'
    static_configs:
      - targets: ['localhost:9121']
        labels:
          cluster: 'vulcan-primary'
          environment: 'production'
EOF

# Reload Prometheus
sudo systemctl reload prometheus

# Import Grafana dashboards
# - Redis Overview Dashboard
# - Redis Keys Dashboard
# - Redis Performance Dashboard
```

---

## 📊 Monitoring Dashboard

### Key Metrics to Monitor

#### Redis Server Metrics
- **Memory Usage**: Current vs. maxmemory
- **Hit Rate**: Cache efficiency
- **Eviction Rate**: Memory pressure indicator
- **Latency**: P50, P95, P99 percentiles
- **Commands/sec**: Throughput
- **Connected Clients**: Connection pool usage
- **Keyspace**: Total keys by database

#### Key Pattern Metrics
- **Pack Keys**: Count and memory usage
- **Bloom Filters**: Hit rate and effectiveness
- **Manifests**: Cache efficiency
- **Document Cache**: Hit rate and size
- **Negative Cache**: Hit rate (should be low)

#### Performance Metrics
- **Slow Queries**: Commands > 10ms
- **Network I/O**: Bytes in/out
- **Replication Lag**: Master-replica delay
- **Fork Duration**: RDB/AOF impact

### Grafana Dashboards

```
Dashboard 1: Redis Overview
- Memory usage trend
- Commands per second
- Hit rate percentage
- Eviction rate
- Connected clients
- Keyspace size

Dashboard 2: Redis Keys
- Key count by pattern
- Memory usage by pattern
- TTL distribution
- Access frequency heatmap
- Eviction events

Dashboard 3: Redis Performance
- Latency percentiles (P50, P95, P99)
- Command latency by type
- Slow log analysis
- Network throughput
- Replication lag
```

---

## 🔧 Maintenance Tasks

### Daily Tasks
- [ ] Review slow query log
- [ ] Check memory usage trends
- [ ] Verify cache hit rates
- [ ] Monitor eviction rates
- [ ] Check replication lag

### Weekly Tasks
- [ ] Review key size distribution
- [ ] Analyze access patterns
- [ ] Optimize TTL policies
- [ ] Review security logs
- [ ] Update documentation

### Monthly Tasks
- [ ] Performance benchmarking
- [ ] Capacity planning review
- [ ] Security audit
- [ ] Backup verification
- [ ] Disaster recovery testing

---

## 🐛 Troubleshooting

### High Memory Usage
```bash
# Check memory usage by key pattern
redis-cli --bigkeys

# Analyze memory
redis-cli MEMORY DOCTOR

# Check fragmentation
redis-cli INFO memory | grep fragmentation

# Solution: Increase maxmemory, enable active defragmentation, adjust TTLs
```

### Low Cache Hit Rate
```bash
# Check hit rate
redis-cli INFO stats | grep keyspace

# Analyze access patterns
redis-cli --hotkeys

# Solution: Increase TTLs, pre-warm cache, optimize queries
```

### High Eviction Rate
```bash
# Check eviction stats
redis-cli INFO stats | grep evicted

# Monitor eviction events
redis-cli --csv --stat | grep evicted

# Solution: Increase memory, reduce TTLs, optimize data structures
```

### Slow Queries
```bash
# Get slow log
redis-cli SLOWLOG GET 10

# Monitor latency
redis-cli --latency

# Solution: Optimize commands, use pipelining, add indexes
```

### Connection Errors
```bash
# Check connections
redis-cli INFO clients

# Monitor connections
redis-cli CLIENT LIST

# Solution: Increase maxclients, optimize connection pool, check network
```

---

## 📈 Performance Benchmarks

### Target Performance Metrics

```
Operations Per Second:
- GET: 100,000+ ops/sec
- SET: 90,000+ ops/sec
- INCR: 100,000+ ops/sec
- LPUSH: 90,000+ ops/sec
- SADD: 100,000+ ops/sec
- HSET: 90,000+ ops/sec

Latency Targets:
- P50: <5ms
- P95: <15ms
- P99: <25ms

Cache Performance:
- Hit Rate: >85%
- Eviction Rate: <1000/sec
- Memory Efficiency: >80%
```

### Benchmark Scripts

```bash
# Basic benchmark
redis-benchmark -t get,set -n 1000000 -q

# Realistic workload
redis-benchmark -t get,set,lpush,sadd -n 1000000 -c 50 -q

# Pipeline benchmark
redis-benchmark -t get,set -n 1000000 -P 16 -q

# Custom benchmark
redis-benchmark -t get -n 1000000 -d 1024 -q
```

---

## 🔐 Security Best Practices

### Configuration Hardening
- ✅ Rename/disable dangerous commands
- ✅ Enable authentication (requirepass)
- ✅ Use ACL for fine-grained access
- ✅ Enable TLS/SSL
- ✅ Bind to specific interfaces
- ✅ Use firewall rules
- ✅ Regular security updates

### Operational Security
- ✅ Monitor access logs
- ✅ Regular security audits
- ✅ Encrypt backups
- ✅ Secure key storage
- ✅ Network segmentation
- ✅ Rate limiting
- ✅ DDoS protection

### Compliance
- ✅ GDPR: Data encryption, access controls, audit logging
- ✅ HIPAA: Encryption, access controls, audit trails
- ✅ SOC2: Monitoring, logging, security controls
- ✅ PCI-DSS: Encryption, access controls, regular audits

---

## 📞 Support and Resources

### Internal Support
- **Email**: redis-support@vulcanami.io
- **Slack**: #redis-support
- **On-Call**: redis-oncall@vulcanami.io
- **Documentation**: https://docs.vulcanami.io/redis

### External Resources
- **Redis Documentation**: https://redis.io/docs/
- **Redis Exporter**: https://github.com/oliver006/redis_exporter
- **Redis Best Practices**: https://redis.io/docs/manual/patterns/
- **Community**: https://redis.io/community/

### Training and Certification
- Redis University: https://university.redis.com/
- Redis Certified Developer: https://redis.com/certification/

---

## 📝 Change Log

### Version 4.6.0 (2025-11-14)
- ✨ Complete configuration overhaul
- ✨ 6-tier key organization system
- ✨ Comprehensive monitoring and alerting
- ✨ Enhanced security hardening
- ✨ Lifecycle management automation
- ✨ Performance optimization
- ✨ Cost optimization strategies
- ✨ Detailed documentation

### Version 4.5.0 (Previous)
- Added Bloom filter keys
- Updated manifest TTL
- Basic monitoring

---

## 📄 License

Copyright © 2025 VulcanAMI Team. All rights reserved.

---

## 🎯 Quick Start Checklist

- [ ] Review all three configuration files
- [ ] Update passwords and sensitive values
- [ ] Adjust memory limits for your environment
- [ ] Configure monitoring endpoints
- [ ] Set up Prometheus scraping
- [ ] Import Grafana dashboards
- [ ] Test in staging environment
- [ ] Deploy to production
- [ ] Verify all metrics are flowing
- [ ] Set up alerting rules
- [ ] Document any custom changes
- [ ] Train team on new configuration

---

**For questions or issues, contact: redis-support@vulcanami.io**