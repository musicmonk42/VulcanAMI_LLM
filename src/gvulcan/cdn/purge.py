"""
CDN Cache Purging with CloudFront Invalidation

This module provides intelligent batch invalidation for CloudFront with support for
rate limiting, retry logic, priority queues, and comprehensive monitoring.
"""

from __future__ import annotations
import time
import boto3
from boto3 import exceptions
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
import hashlib
import threading

logger = logging.getLogger(__name__)

# Configuration constants
MAX_PRIORITY_QUEUE_SIZE = 10000  # Maximum items per priority queue to prevent unbounded growth
RATE_LIMITER_WINDOW_MULTIPLIER = 2  # Multiplier for rate limiter deque size


class PurgePriority(Enum):
    """Priority levels for purge operations"""
    CRITICAL = 0  # Immediate security/compliance issues
    HIGH = 1      # User-facing content updates
    NORMAL = 2    # Regular cache invalidations
    LOW = 3       # Background cleanup


@dataclass
class PurgeRequest:
    """
    Individual purge request with metadata.
    
    Attributes:
        path: CloudFront path pattern to invalidate
        priority: Priority level
        timestamp: When request was created
        reason: Optional reason for purge
        callback: Optional callback function
    """
    path: str
    priority: PurgePriority = PurgePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    reason: Optional[str] = None
    callback: Optional[Callable[[bool, str], None]] = None
    
    def __lt__(self, other: PurgeRequest) -> bool:
        """Compare by priority then timestamp"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class InvalidationResult:
    """Result of a CloudFront invalidation"""
    invalidation_id: str
    paths: List[str]
    status: str
    created_at: datetime
    batch_id: str


@dataclass
class PurgeStats:
    """Statistics for purge operations"""
    total_requested: int = 0
    total_invalidated: int = 0
    total_failed: int = 0
    total_deduplicated: int = 0
    batches_sent: int = 0
    total_bytes_invalidated: int = 0
    average_batch_size: float = 0.0
    last_flush_time: float = 0.0


class PurgeError(Exception):
    """Base exception for purge operations"""
    pass


class CloudFrontRateLimitError(PurgeError):
    """Raised when CloudFront rate limits are hit"""
    pass


class SmartPurger:
    """
    Intelligent CDN cache purger with CloudFront invalidation support.
    
    Features:
    - Batch invalidation with configurable size limits
    - Priority-based queueing
    - Automatic deduplication
    - Rate limiting and backoff
    - Retry logic with exponential backoff
    - Comprehensive metrics and callbacks
    
    Example:
        purger = SmartPurger(distribution_id="E1234567890ABC")
        purger.enqueue("/static/app.js", priority=PurgePriority.HIGH)
        purger.enqueue("/images/*", priority=PurgePriority.NORMAL)
        result = purger.flush()  # Force immediate flush
    """
    
    def __init__(
        self,
        distribution_id: str,
        max_batch_size: int = 3000,
        flush_interval_sec: int = 30,
        max_invalidations_per_hour: int = 100,
        enable_deduplication: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        caller_reference_prefix: str = "vulcan"
    ):
        """
        Initialize smart purger.
        
        Args:
            distribution_id: CloudFront distribution ID
            max_batch_size: Maximum paths per invalidation (CloudFront limit: 3000)
            flush_interval_sec: Time between automatic flushes
            max_invalidations_per_hour: Rate limit for invalidation requests
            enable_deduplication: Whether to deduplicate paths
            max_retries: Maximum retry attempts for failed invalidations
            retry_backoff: Backoff multiplier for retries
            caller_reference_prefix: Prefix for caller references
        """
        self.distribution_id = distribution_id
        self.max_batch_size = min(max_batch_size, 3000)  # CloudFront max
        self.flush_interval_sec = flush_interval_sec
        self.max_invalidations_per_hour = max_invalidations_per_hour
        self.enable_deduplication = enable_deduplication
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.caller_reference_prefix = caller_reference_prefix
        
        # Priority queues for each priority level (bounded to prevent memory issues)
        self.queues = {
            priority: deque(maxlen=MAX_PRIORITY_QUEUE_SIZE) for priority in PurgePriority
        }
        
        # Deduplication tracking
        self.pending_paths: Set[str] = set()
        
        # Rate limiting (keep last hour of timestamps)
        self.invalidation_timestamps: deque = deque(maxlen=max_invalidations_per_hour * RATE_LIMITER_WINDOW_MULTIPLIER)
        self.last_flush = 0.0
        
        # Statistics
        self.stats = PurgeStats()
        
        # CloudFront client
        self.cf_client = boto3.client('cloudfront')
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(
            f"Initialized SmartPurger for distribution {distribution_id}: "
            f"batch_size={max_batch_size}, flush_interval={flush_interval_sec}s"
        )
    
    def _generate_caller_reference(self, paths: List[str]) -> str:
        """Generate unique caller reference for invalidation"""
        timestamp = int(time.time() * 1000)
        path_hash = hashlib.sha256(''.join(sorted(paths)).encode()).hexdigest()[:8]
        return f"{self.caller_reference_prefix}-{timestamp}-{path_hash}"
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        cutoff = now - 3600  # 1 hour ago
        
        # Remove old timestamps
        while self.invalidation_timestamps and self.invalidation_timestamps[0] < cutoff:
            self.invalidation_timestamps.popleft()
        
        return len(self.invalidation_timestamps) < self.max_invalidations_per_hour
    
    def _record_invalidation(self) -> None:
        """Record an invalidation for rate limiting"""
        self.invalidation_timestamps.append(time.time())
    
    def enqueue(
        self,
        path: str,
        priority: PurgePriority = PurgePriority.NORMAL,
        reason: Optional[str] = None,
        callback: Optional[Callable[[bool, str], None]] = None
    ) -> bool:
        """
        Enqueue a path for invalidation.
        
        Args:
            path: CloudFront path pattern (e.g., "/static/*" or "/index.html")
            priority: Priority level for this purge
            reason: Optional reason for purge (for logging)
            callback: Optional callback(success, message) when processed
            
        Returns:
            True if enqueued, False if deduplicated
        """
        with self.lock:
            self.stats.total_requested += 1
            
            # Normalize path
            if not path.startswith('/'):
                path = '/' + path
            
            # Check deduplication
            if self.enable_deduplication and path in self.pending_paths:
                self.stats.total_deduplicated += 1
                logger.debug(f"Path deduplicated: {path}")
                return False
            
            # Create request
            request = PurgeRequest(
                path=path,
                priority=priority,
                reason=reason,
                callback=callback
            )
            
            # Add to appropriate queue
            self.queues[priority].append(request)
            self.pending_paths.add(path)
            
            logger.debug(
                f"Enqueued path: {path} [priority={priority.name}, "
                f"reason={reason or 'N/A'}]"
            )
            
            return True
    
    def enqueue_batch(
        self,
        paths: List[str],
        priority: PurgePriority = PurgePriority.NORMAL,
        reason: Optional[str] = None
    ) -> dict:
        """
        Enqueue multiple paths.
        
        Args:
            paths: List of path patterns
            priority: Priority level
            reason: Optional reason for batch
            
        Returns:
            Dictionary with stats: enqueued, deduplicated
        """
        stats = {'enqueued': 0, 'deduplicated': 0}
        
        for path in paths:
            if self.enqueue(path, priority, reason):
                stats['enqueued'] += 1
            else:
                stats['deduplicated'] += 1
        
        return stats
    
    def _get_next_batch(self) -> List[PurgeRequest]:
        """Get next batch of requests prioritized by priority level"""
        batch = []
        
        # Process by priority order
        for priority in sorted(PurgePriority, key=lambda p: p.value):
            queue = self.queues[priority]
            
            while queue and len(batch) < self.max_batch_size:
                request = queue.popleft()
                batch.append(request)
                
                # Remove from pending set
                if request.path in self.pending_paths:
                    self.pending_paths.discard(request.path)
        
        return batch
    
    def _create_invalidation(
        self,
        paths: List[str],
        caller_reference: str
    ) -> InvalidationResult:
        """
        Create CloudFront invalidation.
        
        Args:
            paths: List of path patterns to invalidate
            caller_reference: Unique reference for this invalidation
            
        Returns:
            InvalidationResult with details
            
        Raises:
            PurgeError: If invalidation fails
        """
        try:
            response = self.cf_client.create_invalidation(
                DistributionId=self.distribution_id,
                InvalidationBatch={
                    'Paths': {
                        'Quantity': len(paths),
                        'Items': paths
                    },
                    'CallerReference': caller_reference
                }
            )
            
            invalidation = response['Invalidation']
            
            return InvalidationResult(
                invalidation_id=invalidation['Id'],
                paths=paths,
                status=invalidation['Status'],
                created_at=invalidation['CreateTime'],
                batch_id=caller_reference
            )
        
        except self.cf_client.exceptions.TooManyInvalidationsInProgress as e:
            raise CloudFrontRateLimitError("Too many invalidations in progress") from e
        
        except Exception as e:
            raise PurgeError(f"Invalidation failed: {e}") from e
    
    def _flush_batch(self, batch: List[PurgeRequest]) -> Optional[InvalidationResult]:
        """Flush a single batch with retry logic"""
        if not batch:
            return None
        
        paths = [req.path for req in batch]
        caller_reference = self._generate_caller_reference(paths)
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                # Check rate limit
                if not self._check_rate_limit():
                    logger.warning("Rate limit reached, delaying invalidation")
                    time.sleep(60)  # Wait a bit
                    continue
                
                # Create invalidation
                result = self._create_invalidation(paths, caller_reference)
                
                # Record for rate limiting
                self._record_invalidation()
                
                # Update stats
                self.stats.total_invalidated += len(paths)
                self.stats.batches_sent += 1
                
                # Call callbacks
                for req in batch:
                    if req.callback:
                        try:
                            req.callback(True, f"Invalidated: {result.invalidation_id}")
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                
                logger.info(
                    f"Invalidation created: {result.invalidation_id} "
                    f"({len(paths)} paths, status={result.status})"
                )
                
                return result
            
            except CloudFrontRateLimitError:
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff ** attempt)
                else:
                    raise
            
            except Exception as e:
                logger.error(f"Invalidation failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff ** attempt)
                else:
                    # Final failure - call error callbacks
                    self.stats.total_failed += len(paths)
                    for req in batch:
                        if req.callback:
                            try:
                                req.callback(False, str(e))
                            except Exception:
                                pass
                    raise
        
        return None
    
    def maybe_flush(self) -> List[InvalidationResult]:
        """
        Conditionally flush based on batch size or time interval.
        
        Returns:
            List of invalidation results
        """
        with self.lock:
            now = time.time()
            total_pending = sum(len(q) for q in self.queues.values())
            
            should_flush = (
                total_pending >= self.max_batch_size or
                (total_pending > 0 and (now - self.last_flush) >= self.flush_interval_sec)
            )
            
            if not should_flush:
                return []
            
            return self.flush()
    
    def flush(self, force: bool = False) -> List[InvalidationResult]:
        """
        Force flush all pending invalidations.
        
        Args:
            force: If True, ignore rate limits (use carefully)
            
        Returns:
            List of invalidation results
        """
        with self.lock:
            results = []
            
            while True:
                batch = self._get_next_batch()
                if not batch:
                    break
                
                try:
                    result = self._flush_batch(batch)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Batch flush failed: {e}")
                    # Re-queue failed requests with lower priority
                    for req in batch:
                        if req.priority != PurgePriority.LOW:
                            req.priority = PurgePriority.NORMAL
                        self.queues[req.priority].append(req)
                        self.pending_paths.add(req.path)
            
            self.last_flush = time.time()
            self.stats.last_flush_time = self.last_flush
            
            if results:
                logger.info(f"Flushed {len(results)} invalidation batches")
            
            return results
    
    def get_stats(self) -> PurgeStats:
        """Get current purge statistics"""
        with self.lock:
            if self.stats.batches_sent > 0:
                self.stats.average_batch_size = (
                    self.stats.total_invalidated / self.stats.batches_sent
                )
            return self.stats
    
    def get_queue_size(self) -> dict:
        """Get size of each priority queue"""
        with self.lock:
            return {
                priority.name: len(queue)
                for priority, queue in self.queues.items()
            }
    
    def clear(self) -> None:
        """Clear all pending invalidations"""
        with self.lock:
            for queue in self.queues.values():
                queue.clear()
            self.pending_paths.clear()
            logger.info("Cleared all pending invalidations")


def create_purger(
    distribution_id: str,
    **kwargs
) -> SmartPurger:
    """
    Convenience function to create a purger.
    
    Args:
        distribution_id: CloudFront distribution ID
        **kwargs: Additional arguments for SmartPurger
        
    Returns:
        Configured SmartPurger instance
    """
    return SmartPurger(distribution_id=distribution_id, **kwargs)