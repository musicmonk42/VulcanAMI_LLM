"""
Production-ready persistent storage backends for Language Evolution Registry
Replaces InMemoryBackend with disk-based storage
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FileSystemBackend:
    """
    File system-based persistent storage backend
    
    Data persists to disk in JSON files organized by key
    Provides reliable persistence with atomic write operations
    """
    
    def __init__(self, storage_dir: str = "data/registry_storage"):
        """
        Initialize file system backend
        
        Args:
            storage_dir: Directory for storing data files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = self.storage_dir / "data"
        self.audit_dir = self.storage_dir / "audit"
        
        self.data_dir.mkdir(exist_ok=True)
        self.audit_dir.mkdir(exist_ok=True)
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger("FileSystemBackend")
        self.logger.info(f"FileSystemBackend initialized at {self.storage_dir}")
    
    def _get_data_path(self, key: str) -> Path:
        """Get path for data file with secure sanitization"""
        # Use SHA-256 hash of key to avoid filesystem injection attacks
        # This prevents: directory traversal, null bytes, control characters, path injection
        import hashlib
        safe_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self.data_dir / f"{safe_key}.json"
    
    def _get_audit_path(self, key: str) -> Path:
        """Get path for audit log file with secure sanitization"""
        # Use SHA-256 hash of key to avoid filesystem injection attacks
        import hashlib
        safe_key = hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self.audit_dir / f"{safe_key}_audit.json"
    
    def load_data(self, key: str) -> Optional[Dict]:
        """Load data for a key"""
        with self._lock:
            try:
                data_path = self._get_data_path(key)
                
                if not data_path.exists():
                    return None
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.logger.debug(f"Loaded data for key: {key}")
                return deepcopy(data)
                
            except Exception as e:
                self.logger.error(f"Failed to load data for key {key}: {e}")
                return None
    
    def save_data(self, key: str, data: Dict) -> str:
        """Save data for a key, returns hash"""
        with self._lock:
            try:
                data_path = self._get_data_path(key)
                
                # Atomic write: write to temp file first, then rename
                temp_path = data_path.with_suffix('.tmp')
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                
                # Atomic rename
                temp_path.replace(data_path)
                
                # Calculate hash
                data_hash = hashlib.sha256(
                    json.dumps(data, sort_keys=True).encode()
                ).hexdigest()
                
                self.logger.debug(f"Saved data for key: {key}")
                return data_hash
                
            except Exception as e:
                self.logger.error(f"Failed to save data for key {key}: {e}")
                raise
    
    def append_chained_record(self, key: str, record: Dict) -> str:
        """Append record to chained log, returns hash"""
        with self._lock:
            try:
                audit_path = self._get_audit_path(key)
                
                # Load existing log
                if audit_path.exists():
                    with open(audit_path, 'r', encoding='utf-8') as f:
                        current_log = json.load(f)
                else:
                    current_log = []
                
                # Calculate previous hash
                previous_hash = current_log[-1]["chain_hash"] if current_log else "0" * 64
                
                # Prepare record with previous hash
                record_for_hash = deepcopy(record)
                record_for_hash["previous_hash"] = previous_hash
                
                # Calculate current hash
                current_record_hash = hashlib.sha256(
                    json.dumps(record_for_hash, sort_keys=True).encode()
                ).hexdigest()
                
                # Add hashes to record
                record["previous_hash"] = previous_hash
                record["chain_hash"] = current_record_hash
                
                # Append to log
                current_log.append(deepcopy(record))
                
                # Atomic write
                temp_path = audit_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(current_log, f, indent=2, sort_keys=True)
                
                temp_path.replace(audit_path)
                
                self.logger.debug(f"Appended chained record for key: {key}")
                return current_record_hash
                
            except Exception as e:
                self.logger.error(f"Failed to append record for key {key}: {e}")
                raise
    
    def get_chained_log(self, key: str) -> List[Dict]:
        """Get all records from chained log"""
        with self._lock:
            try:
                audit_path = self._get_audit_path(key)
                
                if not audit_path.exists():
                    return []
                
                with open(audit_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
                
                return deepcopy(records)
                
            except Exception as e:
                self.logger.error(f"Failed to get chained log for key {key}: {e}")
                return []
    
    def verify_chained_log_integrity(self, key: str) -> bool:
        """Verify chain integrity"""
        with self._lock:
            try:
                records = self.get_chained_log(key)
                
                if not records:
                    return True
                
                previous_hash = "0" * 64
                
                for i, record in enumerate(records):
                    # Check previous hash matches
                    if record.get("previous_hash") != previous_hash:
                        self.logger.error(
                            f"Chain broken at record {i} for key {key}: "
                            f"previous hash mismatch"
                        )
                        return False
                    
                    # Calculate expected hash
                    record_copy = deepcopy(record)
                    record_copy.pop("chain_hash", None)
                    
                    calculated = hashlib.sha256(
                        json.dumps(record_copy, sort_keys=True).encode()
                    ).hexdigest()
                    
                    # Check hash matches
                    if calculated != record.get("chain_hash"):
                        self.logger.error(
                            f"Chain broken at record {i} for key {key}: "
                            f"hash mismatch"
                        )
                        return False
                    
                    previous_hash = record.get("chain_hash")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to verify chain integrity for key {key}: {e}")
                return False
    
    def query_by_prefix(self, prefix: str) -> List[str]:
        """Query keys by prefix"""
        with self._lock:
            try:
                keys = []
                
                for data_file in self.data_dir.glob("*.json"):
                    key = data_file.stem  # Get filename without extension
                    # Restore original key format
                    key = key.replace("_", "/")
                    
                    if key.startswith(prefix):
                        keys.append(key)
                
                return keys
                
            except Exception as e:
                self.logger.error(f"Failed to query by prefix {prefix}: {e}")
                return []


class SQLiteBackend:
    """
    SQLite-based persistent storage backend
    
    Provides efficient querying and better concurrency control
    Suitable for production use with moderate scale
    """
    
    def __init__(self, db_path: str = "data/registry.db"):
        """
        Initialize SQLite backend
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger("SQLiteBackend")
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"SQLiteBackend initialized at {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Create audit log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    record TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    chain_hash TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # Create index for efficient queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_key_prefix 
                ON data(key)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_key 
                ON audit_log(key, timestamp)
            ''')
            
            conn.commit()
            conn.close()
    
    def load_data(self, key: str) -> Optional[Dict]:
        """Load data for a key"""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute(
                    'SELECT value FROM data WHERE key = ?',
                    (key,)
                )
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    data = json.loads(result[0])
                    self.logger.debug(f"Loaded data for key: {key}")
                    return deepcopy(data)
                
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to load data for key {key}: {e}")
                return None
    
    def save_data(self, key: str, data: Dict) -> str:
        """Save data for a key, returns hash"""
        with self._lock:
            try:
                import time
                
                # Calculate hash
                data_hash = hashlib.sha256(
                    json.dumps(data, sort_keys=True).encode()
                ).hexdigest()
                
                # Serialize data
                value_json = json.dumps(data, sort_keys=True)
                
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO data (key, value, hash, updated_at)
                    VALUES (?, ?, ?, ?)
                ''', (key, value_json, data_hash, time.time()))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Saved data for key: {key}")
                return data_hash
                
            except Exception as e:
                self.logger.error(f"Failed to save data for key {key}: {e}")
                raise
    
    def append_chained_record(self, key: str, record: Dict) -> str:
        """Append record to chained log, returns hash"""
        with self._lock:
            try:
                import time
                
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Get previous hash
                cursor.execute('''
                    SELECT chain_hash FROM audit_log 
                    WHERE key = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (key,))
                
                result = cursor.fetchone()
                previous_hash = result[0] if result else "0" * 64
                
                # Prepare record with previous hash
                record_for_hash = deepcopy(record)
                record_for_hash["previous_hash"] = previous_hash
                
                # Calculate current hash
                current_record_hash = hashlib.sha256(
                    json.dumps(record_for_hash, sort_keys=True).encode()
                ).hexdigest()
                
                # Add hashes to record
                record["previous_hash"] = previous_hash
                record["chain_hash"] = current_record_hash
                
                # Insert record
                cursor.execute('''
                    INSERT INTO audit_log (key, record, previous_hash, chain_hash, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    key,
                    json.dumps(record, sort_keys=True),
                    previous_hash,
                    current_record_hash,
                    time.time()
                ))
                
                conn.commit()
                conn.close()
                
                self.logger.debug(f"Appended chained record for key: {key}")
                return current_record_hash
                
            except Exception as e:
                self.logger.error(f"Failed to append record for key {key}: {e}")
                raise
    
    def get_chained_log(self, key: str) -> List[Dict]:
        """Get all records from chained log"""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT record FROM audit_log 
                    WHERE key = ? 
                    ORDER BY timestamp ASC
                ''', (key,))
                
                results = cursor.fetchall()
                conn.close()
                
                records = [json.loads(row[0]) for row in results]
                return deepcopy(records)
                
            except Exception as e:
                self.logger.error(f"Failed to get chained log for key {key}: {e}")
                return []
    
    def verify_chained_log_integrity(self, key: str) -> bool:
        """Verify chain integrity"""
        with self._lock:
            try:
                records = self.get_chained_log(key)
                
                if not records:
                    return True
                
                previous_hash = "0" * 64
                
                for i, record in enumerate(records):
                    # Check previous hash matches
                    if record.get("previous_hash") != previous_hash:
                        self.logger.error(
                            f"Chain broken at record {i} for key {key}: "
                            f"previous hash mismatch"
                        )
                        return False
                    
                    # Calculate expected hash
                    record_copy = deepcopy(record)
                    record_copy.pop("chain_hash", None)
                    
                    calculated = hashlib.sha256(
                        json.dumps(record_copy, sort_keys=True).encode()
                    ).hexdigest()
                    
                    # Check hash matches
                    if calculated != record.get("chain_hash"):
                        self.logger.error(
                            f"Chain broken at record {i} for key {key}: "
                            f"hash mismatch"
                        )
                        return False
                    
                    previous_hash = record.get("chain_hash")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to verify chain integrity for key {key}: {e}")
                return False
    
    def query_by_prefix(self, prefix: str) -> List[str]:
        """Query keys by prefix"""
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT key FROM data 
                    WHERE key LIKE ?
                ''', (f"{prefix}%",))
                
                results = cursor.fetchall()
                conn.close()
                
                keys = [row[0] for row in results]
                return keys
                
            except Exception as e:
                self.logger.error(f"Failed to query by prefix {prefix}: {e}")
                return []


__all__ = ["FileSystemBackend", "SQLiteBackend"]
