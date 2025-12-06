# VulcanAMI Data Quality System (DQS)

**Version**: 2.0.0  
**Status**: Production Ready  
**License**: Proprietary

## Overview

The VulcanAMI Data Quality System (DQS) is an enterprise-grade, multi-dimensional data quality classification and scoring framework designed to ensure high-quality data throughout machine learning pipelines and production systems.

### Key Features

- ✅ **8 Quality Dimensions**: Comprehensive multi-dimensional scoring
- 🔒 **Advanced PII Detection**: ML-powered privacy protection
- 📊 **Graph Analysis**: Complete connectivity and relationship metrics
- 🔄 **Automated Remediation**: Self-healing data quality
- ⏰ **Flexible Scheduling**: 10+ preconfigured rescoring strategies
- 📈 **Real-time Monitoring**: Prometheus metrics and Grafana dashboards
- 🛡️ **GDPR Compliant**: Built-in audit trails and compliance
- ⚡ **High Performance**: Distributed processing with Redis caching

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/vulcanami/dqs.git
cd dqs

# Run installation script
chmod +x setup_dqs.sh
./setup_dqs.sh

# Activate virtual environment
source /opt/vulcanami/dqs/venv/bin/activate

# Verify installation
python dqs_classifier.py
```

### Basic Usage

```python
from dqs_classifier import DataQualityClassifier

# Initialize classifier
classifier = DataQualityClassifier()

# Score data
data = {
    "title": "Machine Learning Research",
    "content": "High-quality research document",
    "created_at": "2025-11-14T00:00:00Z"
}

score = classifier.classify(data, "json", {"source": "arxiv.org"})

# View results
print(f"Score: {score.overall_score:.3f}")
print(f"Category: {score.category}")
print(f"Action: {score.action}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Ingestion                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Data Quality Classifier (8 Dimensions)        │
│  PII Detection • Graph Analysis • Syntactic Validation   │
│  Semantic Validity • Data Freshness • Source Credibility│
│  Consistency • Completeness                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Score Aggregation                       │
│  Weighted Average • Multi-label Classification           │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
     Accept/Warn          Quarantine/Reject
          │                     │
          ▼                     ▼
    Production            Automated Remediation
                               ↓
                          Manual Review
```

## Quality Dimensions

### 1. PII Confidence (15%)
Detects and scores personally identifiable information using transformer models, spaCy NER, and regex patterns.

### 2. Graph Completeness (20%)
Evaluates graph structure, connectivity, node/edge coverage, and relationship density.

### 3. Syntactic Completeness (15%)
Validates format compliance, schema conformance, and structural integrity.

### 4. Semantic Validity (15%)
Checks logical consistency, domain validity, and referential integrity using embeddings.

### 5. Data Freshness (10%)
Scores temporal relevance using exponential decay (90-day half-life).

### 6. Source Credibility (10%)
Evaluates source trustworthiness with whitelist/blacklist and reputation tracking.

### 7. Consistency Score (10%)
Detects internal conflicts, cross-dataset inconsistencies, and statistical outliers.

### 8. Completeness Score (5%)
Measures field presence, value density, and data coverage.

## Classification Categories

| Score | Category | Action | Priority |
|-------|----------|--------|----------|
| 0.90-1.00 | Excellent | Accept | High |
| 0.75-0.90 | Good | Accept | Normal |
| 0.60-0.75 | Fair | Warn | Normal |
| 0.40-0.60 | Poor | Quarantine | Low |
| 0.00-0.40 | Unacceptable | Reject | Critical |

## Rescoring Schedules

- **Full Rescore**: Weekly (Sunday 03:00 UTC)
- **Incremental**: Daily (02:00 UTC)
- **Priority**: Every 6 hours
- **Stale Data**: Weekly (Monday 04:00 UTC)
- **Low Score**: Daily (01:00 UTC)
- **Model Drift**: Monthly (1st, 05:00 UTC)
- **Random Sample**: Every 12 hours
- **Source-Based**: Weekly (Tuesday 06:00 UTC)
- **Remediation Validation**: Hourly (:30)
- **PII Revalidation**: Weekly (Wednesday 07:00 UTC)

## Files Included

### Core Components
- **`classifier.json`**: Comprehensive classifier configuration (1000+ lines)
- **`rescore_cron.json`**: Scheduling system configuration (650+ lines)
- **`dqs_classifier.py`**: Main classification engine (800+ lines)
- **`dqs_rescore.py`**: Rescore orchestrator (500+ lines)

### Documentation
- **`DQS_DOCUMENTATION.md`**: Complete system documentation (1500+ lines)
- **`README.md`**: This file

### Utilities
- **`setup_dqs.sh`**: Automated installation script
- **`dqs_test_suite.py`**: Comprehensive test suite
- **`rescore_cron.crontab`**: Crontab configuration

## Configuration

### Environment Variables

```bash
# Database
export POSTGRES_HOST=postgres
export POSTGRES_PORT=5432
export POSTGRES_DB=vulcanami
export POSTGRES_USER=dqs
export POSTGRES_PASSWORD=dqs_password

# Redis
export REDIS_HOST=redis
export REDIS_PORT=6379
export REDIS_DB=1

# Paths
export DQS_CONFIG=/etc/dqs/classifier.json
export RESCORE_CONFIG=/etc/dqs/rescore_cron.json
```

### Classifier Configuration

Edit `/etc/dqs/classifier.json` to customize:
- Dimension weights
- Score thresholds
- PII detection models
- Remediation strategies
- Monitoring settings

### Rescore Configuration

Edit `/etc/dqs/rescore_cron.json` to customize:
- Schedule timing
- Batch sizes
- Concurrency levels
- Notification channels
- Strategy filters

## Monitoring

### Prometheus Metrics

```
# Classifier
dqs_classifications_total{category="excellent|good|fair|poor|unacceptable"}
dqs_actions_total{action="accept|warn|quarantine|reject"}
dqs_dimension_score{dimension="pii_confidence|graph_completeness|..."}
dqs_classification_duration_seconds

# Rescore
dqs_rescore_items_processed{schedule="...", action="..."}
dqs_rescore_items_failed{schedule="..."}
dqs_rescore_processing_seconds{schedule="..."}
dqs_rescore_score_distribution{schedule="..."}
```

### Endpoints

- **Metrics**: `http://localhost:9145/metrics` (classifier)
- **Metrics**: `http://localhost:9146/metrics` (rescore)
- **Health**: `http://localhost:8080/health`

## Command Line Interface

### Classifier

```bash
# Test classifier
python dqs_classifier.py

# With custom config
python dqs_classifier.py --config /path/to/classifier.json
```

### Rescore Orchestrator

```bash
# List schedules
python dqs_rescore.py list

# Check status
python dqs_rescore.py status

# Run schedule
python dqs_rescore.py run --schedule full_rescore

# Dry run
python dqs_rescore.py run --schedule incremental_rescore --dry-run
```

## Testing

```bash
# Run full test suite
python dqs_test_suite.py

# Run specific tests
python -m unittest dqs_test_suite.TestDataQualityClassifier
python -m unittest dqs_test_suite.TestPerformance

# With verbose output
python dqs_test_suite.py -v
```

## Deployment

### Docker

```dockerfile
FROM python:3.10.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools

# Copy files
COPY . /app
WORKDIR /app

# Install Python packages
RUN pip install -r requirements.txt

# Run classifier
CMD ["python", "dqs_classifier.py"]
```

### Kubernetes

```bash
# Deploy classifier
kubectl apply -f k8s/dqs-classifier-deployment.yaml

# Deploy rescore cron jobs
kubectl apply -f k8s/dqs-rescore-cronjob.yaml

# Check status
kubectl get pods -n dqs
kubectl logs -f deployment/dqs-classifier -n dqs
```

### Systemd

```bash
# Start service
sudo systemctl start dqs-classifier

# Enable on boot
sudo systemctl enable dqs-classifier

# View logs
sudo journalctl -u dqs-classifier -f
```

## Performance

### Benchmarks

- **Classification Speed**: 50-100 items/second (without GPU)
- **Classification Speed**: 200-500 items/second (with GPU)
- **Memory Usage**: ~2-4 GB per worker
- **Cache Hit Rate**: 85-95% (with Redis)
- **Database Throughput**: 10,000+ inserts/second

### Optimization Tips

1. Enable Redis caching for 10x speedup
2. Use GPU for ML models (3-5x speedup)
3. Increase batch sizes for better throughput
4. Scale horizontally with multiple workers
5. Tune PostgreSQL for high write loads

## Troubleshooting

### Low Scores

```sql
-- Check dimension breakdown
SELECT 
    AVG((dimension_scores->>'pii_confidence')::float) as avg_pii,
    AVG((dimension_scores->>'graph_completeness')::float) as avg_graph
FROM dqs.quality_scores
WHERE last_scored_at > NOW() - INTERVAL '1 day';
```

### High PII Detection

```sql
-- Identify PII sources
SELECT 
    metadata->>'source' as source,
    COUNT(*) as pii_count
FROM dqs.quality_scores
WHERE 'contains_pii' = ANY(labels)
GROUP BY source
ORDER BY pii_count DESC;
```

### Slow Processing

```bash
# Check metrics
curl localhost:9146/metrics | grep processing

# Analyze database
EXPLAIN ANALYZE SELECT * FROM dqs.quality_scores 
WHERE last_scored_at < NOW() - INTERVAL '24 hours';
```

## Security

### PII Handling
- All PII detections are logged and audited
- Optional automatic redaction
- Encryption at rest for sensitive data

### Access Control
- Role-based access to audit logs
- API authentication required
- Database user separation

### Compliance
- GDPR-compliant audit trails (7-year retention)
- Right to erasure support
- Data lineage tracking

## Support

### Documentation
- Full Documentation: `DQS_DOCUMENTATION.md`
- API Reference: https://docs.vulcanami.io/dqs/api
- Architecture Guide: https://docs.vulcanami.io/dqs/architecture

### Community
- GitHub: https://github.com/vulcanami/dqs
- Slack: #data-quality
- Email: data-quality@vulcanami.io

### Commercial Support
- Enterprise Support: support@vulcanami.io
- Training: training@vulcanami.io
- Consulting: consulting@vulcanami.io

## License

Copyright © 2025 VulcanAMI. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited.

## Changelog

### Version 2.0.0 (2025-11-14)
- Complete rewrite with 8 quality dimensions
- Advanced PII detection with ML models
- Comprehensive graph analysis
- 10 rescoring strategies
- Prometheus monitoring integration
- GDPR compliance features
- Automated remediation system
- High-performance caching

### Version 1.0.0 (2024-06-01)
- Initial release
- Basic scoring (3 dimensions)
- Simple cron scheduling

## Roadmap

### Q1 2026
- Deep learning quality scoring models
- Active learning feedback loop
- Federated quality assessment
- Real-time streaming support

### Q2 2026
- Advanced ML model deployment
- Custom dimension plugins
- Multi-tenancy support
- Edge computing integration

## Acknowledgments

Built with:
- Python 3.10.11
- PostgreSQL 14
- Redis 6
- PyTorch
- Transformers (HuggingFace)
- spaCy
- NetworkX
- Prometheus

---

**VulcanAMI Data Quality System** - Ensuring Data Excellence at Scale