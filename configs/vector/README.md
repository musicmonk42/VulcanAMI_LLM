# Vector Database Configurations

This directory contains configurations for vector databases used in VulcanAMI's semantic search and embedding storage.

## Contents

### milvus/
Configuration files for Milvus vector database.

#### Files
- `collections.yaml` - Collection definitions, indexes, and schemas

## Purpose

The vector database configurations support:
- **Semantic search**: Fast similarity search over embeddings
- **Knowledge retrieval**: Efficient lookup of related concepts and information
- **Memory systems**: Storage and retrieval for episodic and semantic memory
- **Concept grounding**: Mapping between symbolic and subsymbolic representations

## Milvus Configuration

The Milvus collections are configured for:
- High-dimensional embedding vectors (typically 768 or 1536 dimensions)
- Multiple index types (IVF_FLAT, HNSW) optimized for different use cases
- Metadata filtering and hybrid search capabilities
- Efficient batch operations

## Collection Schema

Collections typically include:
- **Vector field**: High-dimensional embeddings
- **Metadata fields**: Timestamps, source info, confidence scores
- **Text fields**: Original text for reference
- **Category fields**: Domain, type, grounding status

## Usage

The collections are automatically created when the system initializes. Manual management:

```python
from pymilvus import connections, Collection

# Connect to Milvus
connections.connect(host="localhost", port="19530")

# Access collection
collection = Collection("concept_embeddings")
```

## Performance Tuning

Key parameters for optimization:
- `nlist`: Number of clusters for IVF index (typically √n of vectors)
- `m`: Number of connections for HNSW (balance between speed and accuracy)
- `ef_construction`: Build-time parameter for HNSW
- `search_k`: Runtime search parameter

## Scaling

The vector database can scale horizontally:
- Multiple Milvus nodes for distributed storage
- Collection sharding for large datasets
- Read replicas for query load balancing

## Backup and Recovery

- Regular snapshots of collection data
- Export/import functionality for migration
- Point-in-time recovery capabilities

## Monitoring

Monitor these metrics:
- Query latency (p50, p95, p99)
- Index build time
- Memory usage
- Query throughput

## Dependencies

- Milvus 2.x
- pymilvus (Python client)
- etcd (metadata storage)
- MinIO or S3 (blob storage)

## Version

Current configuration version: 1.0.0

## References

- [Milvus Documentation](https://milvus.io/docs)
- [Vector Search Best Practices](https://milvus.io/docs/performance_faq.md)
