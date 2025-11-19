VULCAN-AGI Memory Module

Overview

The Memory Module in VULCAN-AGI implements a sophisticated, multi-faceted memory system inspired by human cognition. It supports hierarchical organization, distributed federation, specialized memory types (e.g., episodic, semantic), efficient retrieval with indexing and attention mechanisms, consolidation/optimization strategies, and persistent storage with compression and versioning. The system is designed for scalability, fault tolerance, and adaptability in AI agents, handling diverse data types while managing decay, importance, and access patterns.

Core principles include thread-safety, fallbacks for optional libraries, and extensibility via enums and abstract classes.

Key Features



Hierarchical Memory: Multi-level storage with automatic promotion/consolidation based on access and importance.

Distributed Federation: Peer-to-peer networking with RPC, encryption, and consistency levels (eventual/strong).

Specialized Memories: Episodic (events), Semantic (concepts), Procedural (skills), Working (short-term buffers).

Retrieval and Search: Vector-based (FAISS fallback to NumPy), keyword/hybrid indexing (Whoosh fallback), learned attention.

Consolidation: Strategies like clustering (KMeans/DBSCAN), graph analysis (NetworkX), information-theoretic optimization.

Persistence: File/SQLite-based storage with LZ4/neural compression, versioning, backups, and cleanup.

Tool Selection History: Tracks decisions for better procedural learning and evolution.

Evolution Engine: Genetic algorithms for semantic/procedural refinement, with optional VULCAN integration.

Monitoring: Stats on access, decay, clustering quality, and network health.



Architecture and Components

The module comprises interconnected classes across files, exported via \_\_init\_\_.py:



base.py: Foundational types like Memory, MemoryType, MemoryQuery, BaseMemorySystem, and exceptions.

hierarchical.py: HierarchicalMemory for leveled storage, with MemoryLevel, pattern mining, and tool records (ToolSelectionRecord, ProblemPattern).

distributed.py: DistributedMemory for federated systems, with MemoryFederation, MemoryNode, RPC (RPCClient/Server), and encryption.

persistence.py: MemoryPersistence for storage/compression (NeuralCompressor), MemoryVersionControl for diffs/backups.

retrieval.py: MemorySearch for queries, MemoryIndex (FAISS/NumPy), AttentionMechanism (PyTorch-based).

consolidation.py: MemoryConsolidator with strategies (ConsolidationStrategy), clustering (ClusteringAlgorithm), graph ops.

specialized.py: Subclasses like EpisodicMemory (Episode), SemanticMemory (Concept, EvolutionEngine), ProceduralMemory (Skill), WorkingMemory (buffers, rehearsal).

\_\_init\_\_.py: Exports all key classes for easy import.



The system uses locks for concurrency, deques for LRU/FIFO, heaps for prioritization, and optional libs for advanced features (with fallbacks).

Installation and Dependencies

This module is part of the VULCAN-AGI project. To use it:



Clone the repository (or integrate into your project).

Install required dependencies:

textpip install numpy scikit-learn networkx faiss-cpu torch sentence-transformers whoosh zmq cryptography redis biopython



Core: numpy, time, logging, typing, dataclasses, collections, threading, hashlib, json, pickle, pathlib, enum, abc, re, ast, operator, copy, socket, struct, concurrent.futures, queue, os, shutil, sqlite3, datetime, heapq, lz4.

Optional: sklearn (clustering/PCA), networkx (graphs), faiss (vector search), torch (neural/attention), sentence\_transformers (embeddings), whoosh (text indexing), zmq (RPC), cryptography (encryption), redis (distributed), biopython/pubchempy (domain-specific, e.g., biology).

Fallbacks for all optionals ensure basic functionality.





Import the module:

pythonfrom vulcan.memory import HierarchicalMemory, MemoryType





Usage Example

pythonimport logging

from vulcan.memory import HierarchicalMemory, Memory, MemoryType, MemoryQuery



\# Set up logging

logging.basicConfig(level=logging.INFO)



\# Initialize hierarchical memory

memory\_system = HierarchicalMemory(

&nbsp;   levels=3,  # Short-term, medium-term, long-term

&nbsp;   embedding\_model='all-MiniLM-L6-v2'  # Optional

)



\# Store a memory

content = "Important event: Meeting at 3 PM"

mem = Memory(

&nbsp;   id="mem001",

&nbsp;   type=MemoryType.EPISODIC,

&nbsp;   content=content,

&nbsp;   importance=0.8

)

memory\_system.store(mem)



\# Query memories

query = MemoryQuery(

&nbsp;   content="meeting",

&nbsp;   query\_type="similarity",

&nbsp;   limit=5

)

results = memory\_system.retrieve(query)

print("Retrieved:", \[(r.memory\_id, r.score) for r in results])



\# Consolidate memories

consolidated = memory\_system.consolidate()

print(f"Consolidated {consolidated} memories")



\# Shutdown

memory\_system.shutdown()

Configuration



Capacity/Thresholds: Set in MemoryConfig (e.g., max\_capacity, consolidation\_threshold).

Strategies: Use enums like ConsolidationStrategy.SEMANTIC\_CLUSTERING or CompressionType.NEURAL.

Embedding Model: Specify in HierarchicalMemory (falls back to hash if unavailable).

Distributed: Configure nodes/hosts in MemoryFederation.

Persistence: Set paths and compression in MemoryPersistence.



Notes



Thread Safety: All operations use reentrant locks; supports background threads (e.g., for consolidation, pattern mining).

Error Handling: Custom exceptions (MemoryCapacityException, etc.) and fallbacks (e.g., NumPy for FAISS).

Performance: Efficient with caching, indexing, and batch ops; monitors stats like access counts and clustering scores.

Extensibility: Extend via abstract classes (e.g., custom ClusteringAlgorithm) or enums.

Limitations: No internet/package installs; relies on pre-imported libs. Distributed assumes secure networks if encryption unavailable.



For contributions or issues, refer to the VULCAN-AGI project repository.

