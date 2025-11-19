# VulcanAMI_LLM - Capabilities Summary

**Quick Reference Guide to Repository Capabilities**

> For comprehensive details, see [CODE_AUDIT.md](./CODE_AUDIT.md)

---

## What This Repository Does

VulcanAMI_LLM is a production-grade AI/AGI platform that combines advanced language models with multi-modal reasoning, safety validation, and self-improvement capabilities. It provides both a REST API server for production deployments and a Python library for direct integration.

---

## Core Capabilities

### 🤖 Language Model (LLM)
- **Advanced transformer architecture** - Custom implementation with IR-based execution
- **Multiple generation modes** - Synchronous, asynchronous, and streaming
- **Context-aware generation** - Integrated with world model and memory systems
- **Configurable sampling** - Temperature, top-k, top-p, beam search, speculative decoding
- **Token-level safety** - Each token validated before emission
- **Explainability** - Built-in reasoning traces for all generations

### 🔒 Production API Server
- **RESTful API** - Flask-based with FastAPI support
- **JWT Authentication** - Multi-signature support (Ed25519, RSA, ECDSA)
- **Role-Based Access Control** - Admin, agent, and custom roles
- **Rate Limiting** - Redis-backed with exponential backoff
- **Audit Logging** - Complete audit trail of all operations
- **TLS Enforcement** - Secure communication for sensitive endpoints

### 🧠 Advanced Reasoning
- **Causal Reasoning** - Intervention simulation, counterfactuals, path analysis
- **Analogical Reasoning** - Structural mapping, abstraction, similarity scoring
- **Probabilistic Reasoning** - Bayesian networks, uncertainty quantification
- **Symbolic Reasoning** - Logic provers, constraint solvers, rule-based systems
- **Language Reasoning** - Strategy-based token selection with confidence scoring

### 🌍 World Model
- **State Tracking** - Dynamic world state maintenance
- **Prediction Engine** - Next-state and trajectory forecasting
- **Intervention Manager** - Safe action simulation with rollback
- **Causal Graph** - Discover and track causal relationships
- **Meta-Reasoning** - Self-improvement, goal management, curiosity-driven exploration

### 🛡️ Safety & Governance
- **Multi-layer Validation** - Token-level and sequence-level safety checks
- **Consensus Engine** - Multi-agent agreement for high-stakes decisions
- **Neural Safety** - Adversarial detection, perplexity monitoring
- **Domain Validators** - Custom safety rules per domain
- **Governance Alignment** - Policy compliance and ethical constraints
- **Rollback Capability** - Undo harmful actions with audit trail

### 💾 Memory Systems
- **Hierarchical Context** - Working, episodic, semantic, and procedural memory
- **Causal Context** - Relevant causal relationships for reasoning
- **Autobiographical Memory** - Timeline of experiences and self-concept
- **Vector Storage** - Efficient similarity search for memory retrieval

### 🏋️ Training & Learning
- **Governed Training** - Safety-constrained gradient updates
- **Self-Improvement** - Autonomous capability enhancement
- **Evolution Engine** - Genetic algorithms for configuration optimization
- **Tournament System** - Model comparison with ELO ratings
- **Checkpoint Management** - Save and restore training state

### ⚡ Execution Runtime
- **IR (Intermediate Representation)** - Hardware-independent computation graphs
- **Hardware Dispatch** - CPU, GPU, TPU routing
- **Graph Validation** - Structure, type, and resource checking
- **Async Execution** - Non-blocking operation support
- **Metrics Collection** - Performance and resource tracking

### 🤝 Orchestration
- **Agent Pool Management** - Dynamic agent scaling
- **Task Queue System** - Priority-based scheduling
- **Dependency Resolution** - Handle complex task graphs
- **Collective Intelligence** - Multi-agent coordination
- **Variant Testing** - A/B testing for configurations

---

## API Endpoints

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `GET /` | Service metadata | None |
| `GET /health` | Health check | None |
| `POST /auth/nonce` | Get authentication nonce | None |
| `POST /auth/login` | Login with signed nonce | None |
| `POST /auth/logout` | Revoke token | JWT |
| `POST /registry/bootstrap` | Create first agent | Bootstrap Key |
| `POST /registry/onboard` | Register new agent | Admin |
| `POST /ir/propose` | Submit IR proposal | JWT |
| `GET /audit/logs` | View audit logs | JWT |

---

## Python API

### Basic Generation

```python
from graphix_vulcan_llm import build_llm

# Build LLM
llm = build_llm(config_path="config.json")

# Generate text
result = llm.generate("Hello, world!", max_tokens=128)
print(result.text)
print(f"Generated {len(result.tokens)} tokens in {result.duration_seconds:.2f}s")

# Get explanation
explanation = result.explanation
print(explanation)
```

### Streaming

```python
# Stream tokens
for token in llm.stream("Tell me a story", max_tokens=100):
    print(token, end=" ", flush=True)
```

### Async Generation

```python
import asyncio

async def generate():
    result = await llm.generate_async("Async prompt", max_tokens=50)
    return result.text

text = asyncio.run(generate())
```

### Training

```python
# Train on dataset
training_data = [{"input": "...", "output": "..."}, ...]
logs = llm.train(training_data, epochs=3, batch_size=8)

# Single training step
record = llm.fine_tune_step({"batch": batch_data})
print(f"Loss: {record.loss}")
```

### Status & Monitoring

```python
# Get system status
status = llm.get_status()
print(f"Total tokens: {status['total_tokens_generated']}")
print(f"Sessions: {status['generation_sessions']}")
print(f"Performance: {status['performance']}")

# Health check
health = llm.health_check()
print(f"Status: {health['status']}")

# Cache stats
cache = llm.get_cache_stats()
print(f"Cache hit rate: {cache['utilization']:.1%}")
```

---

## Key Statistics

- **Lines of Code:** ~70,000+ Python
- **Files:** 406 Python files
- **Dependencies:** 390+ Python packages
- **Test Files:** 20+ test suites
- **API Endpoints:** 11+ REST endpoints

---

## Technology Stack

### Core Technologies
- **Python 3.11+** - Primary language
- **PyTorch 2.8** - Neural network backend
- **Flask 3.0** - REST API framework
- **SQLAlchemy 2.0** - Database ORM
- **Redis 5.2** - Caching and rate limiting

### AI/ML Libraries
- Transformers 4.56 - Hugging Face models
- Sentence Transformers 5.1 - Embeddings
- NetworkX 3.3 - Graph algorithms
- PyTorch Geometric 2.6 - Graph neural networks

### Causal Inference
- causal-learn 0.1.4 - Causal discovery
- dowhy 0.13 - Causal inference
- lingam 1.11 - Linear non-Gaussian models
- pgmpy 1.0 - Probabilistic graphical models

### Optimization
- CVXPY 1.4 - Convex optimization
- Optuna 4.5 - Hyperparameter optimization
- Ray 2.49 - Distributed computing

---

## Use Cases

### 1. Safe AI Applications
Build AI applications with built-in safety validation, audit trails, and explainability.

### 2. Multi-Modal Reasoning
Combine neural and symbolic reasoning for complex problem-solving.

### 3. Causal Analysis
Perform causal inference, intervention simulation, and counterfactual reasoning.

### 4. Agent Systems
Deploy multiple AI agents with orchestration and collective intelligence.

### 5. Research Platform
Experiment with novel AI architectures and reasoning approaches.

### 6. Production LLM Service
Deploy a production-grade LLM API with authentication, rate limiting, and monitoring.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/musicmonk42/VulcanAMI_LLM.git
cd VulcanAMI_LLM

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Run API Server

```bash
# Set required environment variable
export JWT_SECRET_KEY="your-strong-secret-key-here"

# Run server
python app.py
```

Server starts on `http://localhost:5000`

### Use Python Library

```python
from graphix_vulcan_llm import build_llm

# Build and use LLM
llm = build_llm()
result = llm.generate("Hello!", max_tokens=50)
print(result.text)
```

---

## Configuration

### Environment Variables

**Required:**
- `JWT_SECRET_KEY` - Strong secret for JWT signing

**Optional:**
- `BOOTSTRAP_KEY` - Bootstrap protection key
- `DB_URI` - Database connection (default: SQLite)
- `REDIS_HOST` / `REDIS_PORT` - Redis connection
- `CORS_ORIGINS` - Allowed CORS origins (comma-separated)
- `MAX_CONTENT_LENGTH_BYTES` - Request size limit (default: 16MB)
- `IR_MAX_BYTES` - IR size limit (default: 2MB)

### Configuration Files

- `config.json` - LLM configuration (transformer, generation, training)
- `pyproject.toml` - Project metadata
- `requirements.txt` - Python dependencies
- `docker-compose.dev.yml` - Development Docker setup

---

## Production Deployment

### Docker

```bash
# Build image
docker build -t vulcan-ami-llm .

# Run container
docker run -p 5000:5000 \
  -e JWT_SECRET_KEY="your-secret" \
  -e DB_URI="postgresql://user:pass@host/db" \
  -e REDIS_HOST="redis-host" \
  vulcan-ami-llm
```

### Recommended Setup

1. **Database:** PostgreSQL with encryption
2. **Cache:** Redis cluster for high availability
3. **Web Server:** Gunicorn or uWSGI behind nginx
4. **Monitoring:** Prometheus + Grafana
5. **Secrets:** Vault or AWS Secrets Manager
6. **Logging:** ELK or Splunk for centralized logs

---

## Security Features

✅ **Authentication:**
- JWT tokens with multiple signature algorithms
- Nonce-based challenge-response
- Token revocation support

✅ **Authorization:**
- Role-based access control (RBAC)
- Trust scores for fine-grained permissions
- Admin privilege separation

✅ **Protection:**
- Rate limiting with exponential backoff
- Request size limits
- SQL injection prevention (ORM)
- XSS prevention (CSP headers)
- CSRF protection

✅ **Audit:**
- Complete audit trail
- Structured logging
- Compliance tracking

---

## Performance

### Throughput
- **Token generation:** Depends on hardware and model size
- **API requests:** Limited by rate limiting (50/hour default)
- **Concurrent requests:** Supports async/threading

### Optimization
- **Caching:** LRU cache for generation results
- **Batching:** Group operations for efficiency
- **Hardware dispatch:** GPU acceleration support
- **Speculative decoding:** Draft-and-verify acceleration

### Scalability
- **Horizontal:** API server is stateless (with proper setup)
- **Vertical:** Benefits from more CPU cores and GPU
- **Distributed:** Agent orchestration supports distribution

---

## Limitations

⚠️ **Current limitations:**
- Self-improvement is experimental
- Some reasoning engines need validation
- Scalability testing incomplete
- IR sandboxing needs implementation
- Documentation could be more comprehensive

⚠️ **Production considerations:**
- Requires production database (PostgreSQL recommended)
- Requires Redis cluster for high availability
- Needs comprehensive monitoring setup
- Security audit recommended before production
- Performance testing recommended for scale

---

## Support & Resources

- **Full Audit:** [CODE_AUDIT.md](./CODE_AUDIT.md) - Comprehensive technical analysis
- **Repository:** https://github.com/musicmonk42/VulcanAMI_LLM
- **Issues:** GitHub Issues for bugs and feature requests

---

## License

Check repository for license information.

---

**Last Updated:** November 19, 2025  
**Version:** Based on commit 6d5ee67
