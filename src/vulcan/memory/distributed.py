"""Distributed memory implementation with federation support"""

import numpy as np
import time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
import threading
import hashlib
import json
import pickle
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import os

# Network communication
try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    logging.warning("ZMQ not available, using basic socket communication")

# Encryption library
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    logging.warning("Cryptography library not available, encryption is disabled.")

# Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Import missing MemoryType
from .base import (
    Memory, MemoryConfig, MemoryQuery, RetrievalResult,
    ConsistencyLevel, BaseMemorySystem, MemoryType
)

logger = logging.getLogger(__name__)

# ============================================================
# RPC CLIENT/SERVER
# ============================================================

class RPCMessage:
    """RPC message format."""
    
    @staticmethod
    def encode(msg_type: str, data: Any) -> bytes:
        """Encode message for transmission."""
        message = {
            'type': msg_type,
            'data': data,
            'timestamp': time.time()
        }
        return pickle.dumps(message)
    
    @staticmethod
    def decode(data: bytes) -> Dict[str, Any]:
        """Decode received message."""
        return pickle.loads(data)

class RPCClient:
    """RPC client for distributed communication."""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.timeout = 5.0  # seconds
        
        if ZMQ_AVAILABLE:
            self.context = zmq.Context()
        else:
            self.context = None
    
    def connect(self, node_id: str, host: str, port: int) -> bool:
        """Connect to a remote node."""
        try:
            if ZMQ_AVAILABLE:
                socket_conn = self.context.socket(zmq.REQ)
                socket_conn.connect(f"tcp://{host}:{port}")
                socket_conn.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
                socket_conn.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
                self.connections[node_id] = socket_conn
            else:
                # Fallback to basic socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((host, port))
                self.connections[node_id] = sock
            
            logger.info(f"Connected to node {node_id} at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {node_id}: {e}")
            return False
    
    def disconnect(self, node_id: str):
        """Disconnect from a node."""
        if node_id in self.connections:
            try:
                if ZMQ_AVAILABLE:
                    self.connections[node_id].close()
                else:
                    self.connections[node_id].close()
                del self.connections[node_id]
            except Exception as e:                logger.debug(f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}")
    
    def send_request(self, node_id: str, msg_type: str, data: Any) -> Optional[Any]:
        """Send request and wait for response."""
        if node_id not in self.connections:
            return None
        
        try:
            message = RPCMessage.encode(msg_type, data)
            
            if ZMQ_AVAILABLE:
                socket_conn = self.connections[node_id]
                socket_conn.send(message)
                response = socket_conn.recv()
            else:
                sock = self.connections[node_id]
                # Send message length first
                msg_len = len(message)
                sock.send(struct.pack('!I', msg_len))
                sock.sendall(message)
                
                # Receive response
                resp_len_data = sock.recv(4)
                if not resp_len_data:
                    return None
                resp_len = struct.unpack('!I', resp_len_data)[0]
                response = sock.recv(resp_len)
            
            result = RPCMessage.decode(response)
            return result.get('data')
            
        except Exception as e:
            logger.error(f"RPC request failed to {node_id}: {e}")
            return None
    
    async def send_request_async(self, node_id: str, msg_type: str, data: Any) -> Optional[Any]:
        """Send request asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.send_request,
            node_id,
            msg_type,
            data
        )
    
    def cleanup(self):
        """Clean up connections."""
        for node_id in list(self.connections.keys()):
            self.disconnect(node_id)
        
        if self.context:
            self.context.term()

class RPCServer:
    """RPC server for handling distributed requests."""
    
    def __init__(self, host: str, port: int, handler: Any):
        self.host = host
        self.port = port
        self.handler = handler
        self.running = False
        self.server_thread = None
        
        if ZMQ_AVAILABLE:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    def start(self):
        """Start the RPC server."""
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"RPC server started on {self.host}:{self.port}")
    
    def _run_server(self):
        """Run the server loop."""
        if ZMQ_AVAILABLE:
            self.socket.bind(f"tcp://{self.host}:{self.port}")
            
            # Use poller instead of NOBLOCK
            poller = zmq.Poller()
            poller.register(self.socket, zmq.POLLIN)
            
            while self.running:
                try:
                    # Poll with timeout instead of NOBLOCK
                    socks = dict(poller.poll(timeout=100))  # 100ms timeout
                    
                    if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                        message = self.socket.recv()
                        request = RPCMessage.decode(message)
                        
                        # Handle request
                        response = self._handle_request(request)
                        
                        # Send response
                        self.socket.send(RPCMessage.encode('response', response))
                        
                except zmq.ZMQError as e:
                    if e.errno == zmq.EAGAIN:
                        continue  # Timeout, normal
                    logger.error(f"ZMQ error: {e}")
                except Exception as e:
                    logger.error(f"Server error: {e}")
        else:
            # Basic socket server
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            
            while self.running:
                try:
                    self.socket.settimeout(1.0)
                    conn, addr = self.socket.accept()
                    
                    # Handle in thread
                    threading.Thread(
                        target=self._handle_connection,
                        args=(conn,),
                        daemon=True
                    ).start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Server error: {e}")
    
    def _handle_connection(self, conn: socket.socket):
        """Handle a single connection."""
        try:
            # Add message size validation
            MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB limit
            
            # Receive message length
            msg_len_data = conn.recv(4)
            if not msg_len_data:
                return
            
            msg_len = struct.unpack('!I', msg_len_data)[0]
            
            # Validate message length
            if msg_len > MAX_MESSAGE_SIZE:
                logger.error(f"Message too large: {msg_len} bytes (max: {MAX_MESSAGE_SIZE})")
                return
            
            # Receive message
            message = conn.recv(msg_len)
            request = RPCMessage.decode(message)
            
            # Handle request
            response = self._handle_request(request)
            
            # Send response
            resp_data = RPCMessage.encode('response', response)
            conn.send(struct.pack('!I', len(resp_data)))
            conn.sendall(resp_data)
            
        except Exception as e:
            logger.error(f"Connection handler error: {e}")
        finally:
            # Always close socket properly
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except Exception as e:                pass  # Socket might already be closed
            try:
                conn.close()
            except Exception as e:                logger.debug(f"{self.__class__.__name__ if hasattr(self, '__class__') else 'Operation'} error: {e}")
    
    def _handle_request(self, request: Dict[str, Any]) -> Any:
        """Handle incoming request."""
        msg_type = request.get('type')
        data = request.get('data')
        
        if msg_type == 'store':
            return self.handler.handle_store(data)
        elif msg_type == 'retrieve':
            return self.handler.handle_retrieve(data)
        elif msg_type == 'delete':
            return self.handler.handle_delete(data)
        elif msg_type == 'search':
            return self.handler.handle_search(data)
        elif msg_type == 'heartbeat':
            return {'status': 'alive', 'timestamp': time.time()}
        else:
            return {'error': f'Unknown message type: {msg_type}'}
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        
        if ZMQ_AVAILABLE and self.context:
            self.socket.close()
            self.context.term()
        else:
            self.socket.close()

# ============================================================
# MEMORY NODE
# ============================================================

@dataclass
class MemoryNode:
    """Node in distributed memory system."""
    node_id: str
    host: str
    port: int
    capacity: int
    
    # Status
    is_active: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    
    # Load metrics
    memory_count: int = 0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Replication info
    replicas: Set[str] = field(default_factory=set)
    primary_for: Set[str] = field(default_factory=set)
    
    def update_heartbeat(self):
        """Update last heartbeat time."""
        self.last_heartbeat = time.time()
    
    def is_healthy(self, timeout: float = 30.0) -> bool:
        """Check if node is healthy."""
        if not self.is_active:
            return False
        return (time.time() - self.last_heartbeat) < timeout

# ============================================================
# MEMORY FEDERATION
# ============================================================

class MemoryFederation:
    """Federation of distributed memory nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.routing_table: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
        
        # Consensus
        self.consensus_protocol = "raft"
        self.leader_id: Optional[str] = None
        
        # Monitoring
        self.monitor_thread = None
        self.monitor_running = False
        self.start_monitoring()
    
    def register_node(self, node: MemoryNode) -> bool:
        """Register new node in federation."""
        with self.lock:
            if node.node_id in self.nodes:
                return False
            
            self.nodes[node.node_id] = node
            self._update_routing_table()
            
            logger.info(f"Registered node {node.node_id}")
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """Remove node from federation."""
        with self.lock:
            if node_id not in self.nodes:
                return False
            
            # Migrate data from node
            self._migrate_from_node(node_id)
            
            del self.nodes[node_id]
            self._update_routing_table()
            
            logger.info(f"Unregistered node {node_id}")
            return True
    
    def get_nodes_for_key(self, key: str, count: int = 3) -> List[str]:
        """Get nodes responsible for a key using consistent hashing."""
        if not self.nodes:
            return []
        
        # Consistent hashing
        key_hash = self._hash_key(key)
        
        # Get healthy nodes sorted by hash distance
        node_distances = []
        for node_id, node in self.nodes.items():
            if node.is_healthy():
                node_hash = self._hash_key(node_id)
                distance = (node_hash - key_hash) % (2**32)
                node_distances.append((node_id, distance))
        
        node_distances.sort(key=lambda x: x[1])
        
        # Return closest nodes
        return [nid for nid, _ in node_distances[:count]]
    
    def elect_leader(self) -> Optional[str]:
        """Elect leader using consensus protocol."""
        if not self.nodes:
            return None
        
        # Simple leader election: node with lowest ID among healthy nodes
        active_nodes = [
            nid for nid, node in self.nodes.items()
            if node.is_healthy()
        ]
        
        if active_nodes:
            self.leader_id = min(active_nodes)
            logger.info(f"Elected leader: {self.leader_id}")
            return self.leader_id
        
        return None
    
    def _update_routing_table(self):
        """Update routing table based on current nodes."""
        self.routing_table.clear()
        
        if not self.nodes:
            return
        
        # Create hash ring with virtual nodes
        for i in range(256):  # 256 virtual nodes for better distribution
            key = f"virtual_{i}"
            nodes = self.get_nodes_for_key(key, count=3)
            self.routing_table[key] = nodes
    
    def _migrate_from_node(self, node_id: str):
        """Migrate data from failing node."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        # Find target nodes for migration
        for memory_id in node.primary_for:
            new_nodes = self.get_nodes_for_key(memory_id, count=3)
            new_nodes = [nid for nid in new_nodes if nid != node_id]
            
            if new_nodes:
                # Trigger migration
                logger.info(f"Migrating {memory_id} from {node_id} to {new_nodes[0]}")
                # In a real system, this would trigger actual data transfer
    
    def _hash_key(self, key: str) -> int:
        """Hash key to integer for consistent hashing."""
        hash_bytes = hashlib.md5(key.encode()).digest()
        return int.from_bytes(hash_bytes[:4], 'big')
    
    def start_monitoring(self):
        """Start monitoring thread."""
        self.monitor_running = True
        
        def monitor_loop():
            while self.monitor_running:
                time.sleep(10)
                with self.lock:
                    # Check node health
                    unhealthy = []
                    for node_id, node in self.nodes.items():
                        if not node.is_healthy():
                            logger.warning(f"Node {node_id} unhealthy")
                            unhealthy.append(node_id)
                    
                    # Remove unhealthy nodes
                    for node_id in unhealthy:
                        self.unregister_node(node_id)
                    
                    # Re-elect leader if needed
                    if self.leader_id:
                        if self.leader_id not in self.nodes or \
                           not self.nodes[self.leader_id].is_healthy():
                            self.elect_leader()
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring thread."""
        self.monitor_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

# ============================================================
# DISTRIBUTED MEMORY
# ============================================================

class DistributedMemory(BaseMemorySystem):
    """Distributed memory system with replication and sharding."""
    
    def __init__(self, config: MemoryConfig, 
                 federation: Optional[MemoryFederation] = None,
                 node_id: Optional[str] = None,
                 host: str = "localhost",
                 port: int = 5555,
                 federation_key: Optional[bytes] = None):
        super().__init__(config)
        
        self.node_id = node_id or f"node_{hash(f'{host}:{port}')}"
        self.host = host
        self.port = port
        
        self.federation = federation or MemoryFederation()
        self.local_storage: Dict[str, Memory] = {}
        self.replicas: Dict[str, Set[str]] = {}  # memory_id -> node_ids
        
        # Use shared federation key
        self.cipher = None
        if ENCRYPTION_AVAILABLE:
            if federation_key:
                # Use provided federation-wide key
                self.cipher = Fernet(federation_key)
                logger.info("Using shared federation encryption key")
            else:
                key = os.getenv('MEMORY_ENCRYPT_KEY')
                if key:
                    self.cipher = Fernet(key.encode())
                    logger.info("Using encryption key from environment")
                else:
                    # Generate ephemeral key and warn about federation issues
                    self.cipher = Fernet(Fernet.generate_key())
                    logger.warning("=" * 60)
                    logger.warning("GENERATED EPHEMERAL ENCRYPTION KEY")
                    logger.warning("This key is NOT shared across federation nodes.")
                    logger.warning("Decryption will fail if memories are replicated.")
                    logger.warning("Set MEMORY_ENCRYPT_KEY or provide federation_key parameter.")
                    logger.warning("=" * 60)

        # Redis client fallback
        self.redis_client: Optional[redis.Redis] = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
                self.redis_client.ping()  # Test conn
                logger.info("Successfully connected to Redis.")
            except Exception:
                logger.warning("Redis server unavailable; operating in local-only/distributed mode.")
                self.redis_client = None

        # Consistency
        self.consistency_level = config.consistency_level
        self.vector_clocks: Dict[str, Dict[str, int]] = {}
        
        # Communication
        self.rpc_client = RPCClient()
        self.rpc_server = RPCServer(host, port, self)
        
        # Start server
        self.rpc_server.start()
        
        # Register self in federation
        self_node = MemoryNode(
            node_id=self.node_id,
            host=host,
            port=port,
            capacity=config.max_long_term
        )
        self.federation.register_node(self_node)
        
        # Connect to other nodes
        self._connect_to_nodes()
    
    def _connect_to_nodes(self):
        """Connect to other nodes in the federation."""
        for node_id, node in self.federation.nodes.items():
            if node_id != self.node_id:
                self.rpc_client.connect(node_id, node.host, node.port)
    
    def store(self, content: Any, **kwargs) -> Memory:
        """Store with replication."""
        # Create memory
        memory = Memory(
            id=self._generate_id(content),
            type=kwargs.get('memory_type', MemoryType.LONG_TERM),
            content=content,
            timestamp=time.time(),
            **kwargs
        )
        
        # FIX: Try Redis as CACHE LAYER only - don't return early
        if self.redis_client:
            try:
                key = f"mem:{memory.id}"
                mem_copy = pickle.loads(pickle.dumps(memory))
                if self.cipher:
                    serialized_content = pickle.dumps(mem_copy.content)
                    mem_copy.content = self.cipher.encrypt(serialized_content)
                    mem_copy.metadata['encrypted'] = True
                
                serialized_memory = pickle.dumps(mem_copy)
                self.redis_client.set(key, serialized_memory, ex=3600)
                logger.debug(f"Cached memory {memory.id} in Redis.")
                # FIX: Continue to local/distributed storage instead of returning
            except Exception as e:
                logger.error(f"Failed to cache in Redis: {e}")
                # Continue anyway

        # Create encrypted copy for storage/replication
        encrypted_memory = pickle.loads(pickle.dumps(memory))
        
        # Encrypt content before storing and replicating
        if self.cipher:
            try:
                serialized_content = pickle.dumps(encrypted_memory.content)
                encrypted_memory.content = self.cipher.encrypt(serialized_content)
                encrypted_memory.metadata['encrypted'] = True
            except Exception as e:
                logger.error(f"Failed to encrypt memory {memory.id}: {e}")
                raise
        
        # Get target nodes for this memory
        target_nodes = self.federation.get_nodes_for_key(
            memory.id,
            count=self.config.replication_factor
        )
        
        # FIX: Always store locally (either as target or fallback for single node)
        if not target_nodes or len(self.federation.nodes) <= 1 or self.node_id in target_nodes:
            self.local_storage[memory.id] = encrypted_memory
            self.federation.nodes[self.node_id].memory_count += 1
            self.federation.nodes[self.node_id].primary_for.add(memory.id)
        
        # Replicate to other target nodes
        if target_nodes and len(target_nodes) > 1:
            success_count = self._replicate_to_nodes(encrypted_memory, target_nodes)
            
            # Check consistency requirements
            if self.consistency_level == ConsistencyLevel.STRONG:
                required = len(target_nodes)
            elif self.consistency_level == ConsistencyLevel.LINEARIZABLE:
                required = (len(target_nodes) + 1) // 2
            else:
                required = 1
            
            if success_count < required:
                raise Exception(f"Failed to achieve required consistency: {success_count}/{required}")
            
            # Track replicas
            self.replicas[memory.id] = set(target_nodes[:success_count])
        else:
            # FIX: Single node or no other targets - still track as replica
            self.replicas[memory.id] = {self.node_id}
        
        self.stats.total_stores += 1
        return memory
    
    def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        """Retrieve with consistency guarantees."""
        start_time = time.time()
        
        # Search local storage first
        local_results = self._search_local(query)
        
        # Check if we need to query other nodes
        if self.consistency_level != ConsistencyLevel.EVENTUAL or len(local_results) < query.limit:
            # Query other nodes
            remote_results = self._search_remote(query)
            
            # Merge results
            all_results = self._merge_results(local_results, remote_results)
        else:
            all_results = local_results
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x[1], reverse=True)
        memories = [m for m, _ in all_results[:query.limit]]
        scores = [s for _, s in all_results[:query.limit]]

        # Decrypt content before returning
        if self.cipher:
            for mem in memories:
                if mem.metadata.get('encrypted') and isinstance(mem.content, bytes):
                    try:
                        decrypted_content = self.cipher.decrypt(mem.content)
                        mem.content = pickle.loads(decrypted_content)
                        mem.metadata['encrypted'] = False
                    except Exception as e:
                        logger.warning(f"Failed to decrypt memory {mem.id}, returning encrypted content. Error: {e}")

        
        return RetrievalResult(
            memories=memories,
            scores=scores,
            query_time_ms=(time.time() - start_time) * 1000,
            total_matches=len(all_results)
        )
    
    def forget(self, memory_id: str) -> bool:
        """Delete with replica cleanup."""
        # Find replicas
        replica_nodes = self.replicas.get(memory_id, set())
        
        # Delete from all replicas
        success_count = 0
        for node_id in replica_nodes:
            if node_id == self.node_id:
                # Delete locally
                if memory_id in self.local_storage:
                    del self.local_storage[memory_id]
                    self.federation.nodes[self.node_id].memory_count -= 1
                    success_count += 1
            else:
                # Delete from remote node
                if self._delete_from_node(memory_id, node_id):
                    success_count += 1
        
        # Clean up tracking
        if memory_id in self.replicas:
            del self.replicas[memory_id]
        
        if self.node_id in self.federation.nodes:
            self.federation.nodes[self.node_id].primary_for.discard(memory_id)
        
        return success_count > 0
    
    def consolidate(self) -> int:
        """Consolidate across distributed nodes."""
        consolidated = 0
        
        # Rebalance if needed
        if self._needs_rebalancing():
            consolidated += self._rebalance_data()
        
        # Clean up orphaned replicas
        consolidated += self._cleanup_orphaned_replicas()
        
        return consolidated
    
    def _replicate_to_nodes(self, memory: Memory, node_ids: List[str]) -> int:
        """Replicate memory to nodes."""
        success_count = 0
        
        for node_id in node_ids:
            if node_id == self.node_id:
                # Already stored locally
                success_count += 1
            else:
                # Send to remote node
                if self._send_to_node(memory, node_id):
                    success_count += 1
        
        return success_count
    
    def _send_to_node(self, memory: Memory, node_id: str) -> bool:
        """Send memory to specific node."""
        try:
            # Prepare memory data for transmission
            # Content is already encrypted if needed
            memory_data = {
                'id': memory.id,
                'type': memory.type.value,
                'content': memory.content,
                'embedding': memory.embedding.tolist() if memory.embedding is not None else None,
                'timestamp': memory.timestamp,
                'importance': memory.importance,
                'metadata': memory.metadata
            }
            
            # Send via RPC
            result = self.rpc_client.send_request(node_id, 'store', memory_data)
            
            if result and result.get('success'):
                logger.debug(f"Successfully sent memory {memory.id} to node {node_id}")
                return True
            else:
                logger.warning(f"Failed to send memory {memory.id} to node {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to node {node_id}: {e}")
            return False
    
    def _delete_from_node(self, memory_id: str, node_id: str) -> bool:
        """Delete memory from specific node."""
        try:
            result = self.rpc_client.send_request(node_id, 'delete', {'memory_id': memory_id})
            
            if result and result.get('success'):
                logger.debug(f"Successfully deleted memory {memory_id} from node {node_id}")
                return True
            else:
                logger.warning(f"Failed to delete memory {memory_id} from node {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting from node {node_id}: {e}")
            return False
    
    def _search_local(self, query: MemoryQuery) -> List[Tuple[Memory, float]]:
        """Search local storage."""
        results = []
        
        for memory in self.local_storage.values():
            # Apply filters
            if self._matches_query(memory, query):
                score = self._compute_score(memory, query)
                if score >= query.threshold:
                    results.append((memory, score))
        
        return results
    
    def _search_remote(self, query: MemoryQuery) -> List[Tuple[Memory, float]]:
        """Search remote nodes using distributed query."""
        remote_results = []
        
        # Prepare query for transmission
        query_data = {
            'query_type': query.query_type,
            'content': query.content,
            'embedding': query.embedding.tolist() if query.embedding is not None else None,
            'filters': query.filters,
            'time_range': query.time_range,
            'limit': query.limit,
            'threshold': query.threshold
        }
        
        # Query each healthy node in parallel
        futures = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for node_id, node in self.federation.nodes.items():
                if node_id != self.node_id and node.is_healthy():
                    future = executor.submit(
                        self.rpc_client.send_request,
                        node_id,
                        'search',
                        query_data
                    )
                    futures.append((node_id, future))
            
            # Collect results
            for node_id, future in futures:
                try:
                    result = future.result(timeout=5.0)
                    if result and 'results' in result:
                        for mem_data, score in result['results']:
                            # Reconstruct memory object
                            memory = self._reconstruct_memory(mem_data)
                            if memory:
                                remote_results.append((memory, score))
                                
                except Exception as e:
                    logger.warning(f"Failed to get search results from {node_id}: {e}")
        
        return remote_results
    
    def _reconstruct_memory(self, mem_data: Dict) -> Optional[Memory]:
        """Reconstruct Memory object from transmitted data."""
        try:
            embedding = None
            if mem_data.get('embedding'):
                embedding = np.array(mem_data['embedding'])
            
            memory = Memory(
                id=mem_data['id'],
                type=MemoryType(mem_data['type']),
                content=mem_data['content'],
                embedding=embedding,
                timestamp=mem_data['timestamp'],
                importance=mem_data.get('importance', 0.5),
                metadata=mem_data.get('metadata', {})
            )
            
            return memory
            
        except Exception as e:
            logger.error(f"Failed to reconstruct memory: {e}")
            return None
    
    def _merge_results(self, local: List[Tuple[Memory, float]], 
                      remote: List[Tuple[Memory, float]]) -> List[Tuple[Memory, float]]:
        """Merge and deduplicate results."""
        seen = {}
        
        # Process all results, keeping highest score for duplicates
        for memory, score in local + remote:
            if memory.id not in seen or seen[memory.id][1] < score:
                seen[memory.id] = (memory, score)
        
        # Convert back to list
        merged = list(seen.values())
        
        return merged
    
    def _matches_query(self, memory: Memory, query: MemoryQuery) -> bool:
        """Check if memory matches query filters."""
        # Time range filter
        if query.time_range:
            start, end = query.time_range
            if not (start <= memory.timestamp <= end):
                return False
        
        # Type filter
        if 'type' in query.filters:
            if memory.type != query.filters['type']:
                return False
        
        # Importance filter
        if 'min_importance' in query.filters:
            if memory.importance < query.filters['min_importance']:
                return False
        
        # Metadata filters
        if 'metadata' in query.filters:
            for key, value in query.filters['metadata'].items():
                if memory.metadata.get(key) != value:
                    return False
        
        return True
    
    def _compute_score(self, memory: Memory, query: MemoryQuery) -> float:
        """Compute relevance score."""
        if query.embedding is not None and memory.embedding is not None:
            # Cosine similarity
            norm_query = query.embedding / (np.linalg.norm(query.embedding) + 1e-10)
            norm_memory = memory.embedding / (np.linalg.norm(memory.embedding) + 1e-10)
            score = float(np.dot(norm_query, norm_memory))
        else:
            # Use salience as score
            score = memory.compute_salience()
        
        return score
    
    def _generate_id(self, content: Any) -> str:
        """Generate unique ID with high-precision timestamp."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        # Use high precision timestamp and add random component to ensure uniqueness
        timestamp = f"{time.time():.9f}"
        random_component = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        node_id = self.node_id
        combined = f"{content_str}_{timestamp}_{random_component}_{node_id}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _needs_rebalancing(self) -> bool:
        """Check if data needs rebalancing."""
        if len(self.federation.nodes) < 2:
            return False
        
        # Check load variance
        loads = [node.memory_count for node in self.federation.nodes.values()]
        if not loads:
            return False
        
        avg_load = sum(loads) / len(loads)
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        
        # Rebalance if variance is high
        threshold = (avg_load * 0.5) ** 2
        return variance > threshold
    
    def _rebalance_data(self) -> int:
        """Rebalance data across nodes."""
        rebalanced = 0
        
        # Calculate target distribution
        total_memories = sum(node.memory_count for node in self.federation.nodes.values())
        target_per_node = total_memories // max(1, len(self.federation.nodes))
        
        # Find overloaded and underloaded nodes
        overloaded = []
        underloaded = []
        
        for node_id, node in self.federation.nodes.items():
            if node.memory_count > target_per_node * 1.2:
                overloaded.append(node_id)
            elif node.memory_count < target_per_node * 0.8:
                underloaded.append(node_id)
        
        # Migrate from overloaded to underloaded
        if overloaded and underloaded and self.node_id in overloaded:
            # Migrate some of our memories
            memories_to_migrate = []
            excess = self.federation.nodes[self.node_id].memory_count - target_per_node
            
            for memory_id in list(self.local_storage.keys())[:excess]:
                memories_to_migrate.append(self.local_storage[memory_id])
            
            # Send to underloaded nodes
            target_idx = 0
            for memory in memories_to_migrate:
                if target_idx < len(underloaded):
                    target_node = underloaded[target_idx]
                    if self._send_to_node(memory, target_node):
                        # Remove from local storage
                        del self.local_storage[memory.id]
                        self.federation.nodes[self.node_id].memory_count -= 1
                        rebalanced += 1
                    
                    target_idx = (target_idx + 1) % len(underloaded)
        
        if rebalanced > 0:
            logger.info(f"Rebalanced {rebalanced} memories")
        
        return rebalanced
    
    def _cleanup_orphaned_replicas(self) -> int:
        """Clean up replicas on failed nodes."""
        cleaned = 0
        
        for memory_id, replica_nodes in list(self.replicas.items()):
            # Check if replicas are healthy
            healthy_replicas = set()
            for node_id in replica_nodes:
                node = self.federation.nodes.get(node_id)
                if node and node.is_healthy():
                    healthy_replicas.add(node_id)
            
            # Re-replicate if needed
            if len(healthy_replicas) < self.config.replication_factor:
                # Find new nodes
                new_nodes = self.federation.get_nodes_for_key(
                    memory_id,
                    count=self.config.replication_factor
                )
                
                # Ensure we have the memory to replicate
                if memory_id in self.local_storage:
                    for node_id in new_nodes:
                        if node_id not in healthy_replicas and node_id != self.node_id:
                            # Create new replica
                            if self._send_to_node(self.local_storage[memory_id], node_id):
                                healthy_replicas.add(node_id)
                                cleaned += 1
                
                self.replicas[memory_id] = healthy_replicas
        
        return cleaned
    
    # RPC Handler Methods
    def handle_store(self, data: Dict) -> Dict:
        """Handle incoming store request."""
        try:
            # Reconstruct memory
            memory = self._reconstruct_memory(data)
            if memory:
                # The content is received encrypted and stored as is.
                self.local_storage[memory.id] = memory
                self.federation.nodes[self.node_id].memory_count += 1
                return {'success': True, 'memory_id': memory.id}
            
            return {'success': False, 'error': 'Failed to reconstruct memory'}
            
        except Exception as e:
            logger.error(f"Store handler error: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_retrieve(self, data: Dict) -> Dict:
        """Handle incoming retrieve request."""
        try:
            memory_id = data.get('memory_id')
            if memory_id and memory_id in self.local_storage:
                memory = self.local_storage[memory_id]
                # The content is sent back still encrypted.
                # The requesting node is responsible for decryption.
                return {
                    'success': True,
                    'memory': {
                        'id': memory.id,
                        'type': memory.type.value,
                        'content': memory.content,
                        'embedding': memory.embedding.tolist() if memory.embedding is not None else None,
                        'timestamp': memory.timestamp,
                        'importance': memory.importance,
                        'metadata': memory.metadata
                    }
                }
            
            return {'success': False, 'error': 'Memory not found'}
            
        except Exception as e:
            logger.error(f"Retrieve handler error: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_delete(self, data: Dict) -> Dict:
        """Handle incoming delete request."""
        try:
            memory_id = data.get('memory_id')
            if memory_id and memory_id in self.local_storage:
                del self.local_storage[memory_id]
                self.federation.nodes[self.node_id].memory_count -= 1
                return {'success': True}
            
            return {'success': False, 'error': 'Memory not found'}
            
        except Exception as e:
            logger.error(f"Delete handler error: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_search(self, data: Dict) -> Dict:
        """Handle incoming search request."""
        try:
            # Reconstruct query
            query = MemoryQuery(
                query_type=data.get('query_type', 'similarity'),
                content=data.get('content'),
                embedding=np.array(data['embedding']) if data.get('embedding') else None,
                filters=data.get('filters', {}),
                time_range=data.get('time_range'),
                limit=data.get('limit', 10),
                threshold=data.get('threshold', 0.5)
            )
            
            # Search local storage only (no recursive remote search)
            local_results = self._search_local(query)
            
            # Complete results preparation
            results = []
            for memory, score in local_results:
                mem_data = {
                    'id': memory.id,
                    'type': memory.type.value,
                    'content': memory.content,
                    'embedding': memory.embedding.tolist() if memory.embedding is not None else None,
                    'timestamp': memory.timestamp,
                    'importance': memory.importance,
                    'metadata': memory.metadata
                }
                results.append((mem_data, score))
            
            return {
                'success': True,
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Search handler error: {e}")
            return {'success': False, 'error': str(e)}
