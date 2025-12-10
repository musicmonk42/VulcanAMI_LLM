#!/usr/bin/env python3
"""
VulcanAMI Data Quality System - Rescore Orchestrator
Handles scheduled rescoring operations with distributed execution
"""

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

import psycopg2
import redis
from dqs_classifier import DataQualityClassifier, QualityScore
from prometheus_client import (Counter, Gauge, Histogram, push_to_gateway,
                               start_http_server)
from psycopg2.extras import RealDictCursor

# Prometheus metrics
ITEMS_PROCESSED = Counter('dqs_rescore_items_processed', 'Items processed', ['schedule', 'action'])
ITEMS_FAILED = Counter('dqs_rescore_items_failed', 'Items failed', ['schedule'])
PROCESSING_TIME = Histogram('dqs_rescore_processing_seconds', 'Processing time', ['schedule'])
QUEUE_DEPTH = Gauge('dqs_rescore_queue_depth', 'Queue depth', ['schedule'])
SCORE_DISTRIBUTION = Histogram('dqs_rescore_score_distribution', 'Score distribution',
                              ['schedule'], buckets=[0.0, 0.3, 0.4, 0.6, 0.75, 0.9, 1.0])


@dataclass
class RescoreJob:
    """Represents a rescore job"""
    id: str
    schedule_name: str
    strategy: str
    filters: Dict
    batch_size: int
    max_items: Optional[int]
    timeout_hours: int
    priority: str


class RescoreOrchestrator:
    """Orchestrates data quality rescoring operations"""

    def __init__(self, config_path: str = "/etc/dqs/rescore_cron.json",
                 classifier_config: str = "/etc/dqs/classifier.json"):
        """Initialize the orchestrator"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.classifier = DataQualityClassifier(classifier_config)
        self.db_conn = self._setup_database()
        self.redis_client = self._setup_redis()

        # State management
        self.running = False
        self.current_job = None
        self.processed_count = 0
        self.failed_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.info("RescoreOrchestrator initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load rescore configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        log_config = self.config.get('monitoring', {}).get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('DQS.Rescore')

    def _setup_database(self) -> psycopg2.extensions.connection:
        """Setup database connection"""
        source_config = self.config['data_selection']['sources']['primary']
        return psycopg2.connect(
            host=source_config['host'],
            port=source_config['port'],
            database=source_config['database'],
            user='dqs',
            password='',  # Use environment variable
            cursor_factory=RealDictCursor
        )

    def _setup_redis(self) -> redis.Redis:
        """Setup Redis connection"""
        cache_config = self.config['data_selection']['sources']['cache']
        return redis.Redis(
            host=cache_config['host'],
            port=cache_config['port'],
            db=cache_config['db'],
            decode_responses=True
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False

    def execute_schedule(self, schedule_name: str, dry_run: bool = False):
        """Execute a specific schedule"""
        if schedule_name not in self.config['schedules']:
            raise ValueError(f"Unknown schedule: {schedule_name}")

        schedule = self.config['schedules'][schedule_name]

        if not schedule['enabled']:
            self.logger.warning(f"Schedule {schedule_name} is disabled")
            return

        self.logger.info(f"Starting schedule: {schedule_name}")
        self.running = True
        self.current_job = schedule_name

        start_time = time.time()

        try:
            # Send start notification
            if schedule['notifications']['on_start']:
                self._send_notification(
                    f"Starting rescore: {schedule_name}",
                    schedule['notifications']['channels']
                )

            # Create job
            job = RescoreJob(
                id=f"{schedule_name}_{int(time.time())}",
                schedule_name=schedule_name,
                strategy=schedule['strategy'],
                filters=schedule.get('filters', {}),
                batch_size=self.config['strategies'][schedule['strategy']]['batch_size'],
                max_items=schedule.get('max_items'),
                timeout_hours=schedule['timeout_hours'],
                priority=schedule['priority']
            )

            # Execute job
            self._execute_job(job, dry_run)

            # Send completion notification
            duration = time.time() - start_time
            if schedule['notifications']['on_complete']:
                self._send_notification(
                    f"Completed rescore: {schedule_name}\n"
                    f"Processed: {self.processed_count}\n"
                    f"Failed: {self.failed_count}\n"
                    f"Duration: {duration:.1f}s",
                    schedule['notifications']['channels']
                )

            self.logger.info(f"Completed schedule {schedule_name} in {duration:.1f}s")

        except Exception as e:
            self.logger.error(f"Error executing schedule {schedule_name}: {e}")

            if schedule['notifications']['on_failure']:
                self._send_notification(
                    f"Failed rescore: {schedule_name}\n"
                    f"Error: {str(e)}",
                    schedule['notifications']['channels']
                )

            raise
        finally:
            self.running = False
            self.current_job = None

    def _execute_job(self, job: RescoreJob, dry_run: bool = False):
        """Execute a rescore job"""
        self.logger.info(f"Executing job {job.id} with strategy {job.strategy}")

        # Get items to process
        items_iterator = self._get_items_to_rescore(job)

        # Process in batches
        batch = []
        total_processed = 0

        for item in items_iterator:
            if not self.running:
                self.logger.warning("Job interrupted, stopping...")
                break

            batch.append(item)

            if len(batch) >= job.batch_size:
                self._process_batch(batch, job, dry_run)
                total_processed += len(batch)
                batch = []

                # Check max items limit
                if job.max_items and total_processed >= job.max_items:
                    self.logger.info(f"Reached max items limit: {job.max_items}")
                    break

        # Process remaining items
        if batch:
            self._process_batch(batch, job, dry_run)
            total_processed += len(batch)

        self.logger.info(f"Job {job.id} processed {total_processed} items")

    def _get_items_to_rescore(self, job: RescoreJob) -> Iterator[Dict]:
        """Get items to rescore based on job strategy"""
        strategy_config = self.config['strategies'][job.strategy]

        # Build query based on strategy
        query = self._build_query(job)

        self.logger.info(f"Executing query for strategy {job.strategy}")

        with self.db_conn.cursor() as cur:
            cur.execute(query, job.filters)

            # Fetch and yield in batches
            while True:
                rows = cur.fetchmany(job.batch_size)
                if not rows:
                    break

                for row in rows:
                    yield dict(row)

                    if not self.running:
                        break

    def _build_query(self, job: RescoreJob) -> str:
        """Build SQL query based on strategy and filters"""
        strategy = job.strategy
        filters = job.filters

        base_query = """
            SELECT
                id,
                data,
                data_type,
                metadata,
                current_score,
                last_scored_at,
                created_at,
                updated_at
            FROM dqs.quality_scores
        """

        where_clauses = []

        if strategy == "incremental":
            lookback_hours = filters.get('lookback_hours', 24)
            where_clauses.append(
                f"updated_at > NOW() - INTERVAL '{lookback_hours} hours'"
            )

        elif strategy == "priority_based":
            statuses = filters.get('statuses', [])
            if statuses:
                where_clauses.append(f"action IN ({','.join(['%s']*len(statuses))})")

            min_age = filters.get('min_age_hours', 0)
            if min_age > 0:
                where_clauses.append(f"last_scored_at < NOW() - INTERVAL '{min_age} hours'")

        elif strategy == "age_based":
            min_days = filters.get('min_age_days', 30)
            max_days = filters.get('max_age_days', 365)
            where_clauses.append(
                f"last_scored_at BETWEEN NOW() - INTERVAL '{max_days} days' "
                f"AND NOW() - INTERVAL '{min_days} days'"
            )

        elif strategy == "score_based":
            max_score = filters.get('max_score', 0.60)
            where_clauses.append(f"current_score <= {max_score}")

            min_attempts = filters.get('min_rescore_attempts', 1)
            max_attempts = filters.get('max_rescore_attempts', 5)
            where_clauses.append(
                f"rescore_attempts BETWEEN {min_attempts} AND {max_attempts}"
            )

        elif strategy == "random_sample":
            # Random sampling handled separately
            sample_size = job.filters.get('sample_size', 1000)
            return f"{base_query} ORDER BY RANDOM() LIMIT {sample_size}"

        # Apply exclude filters
        if self.config['data_selection']['filters']['exclude_recently_scored']['enabled']:
            min_hours = self.config['data_selection']['filters']['exclude_recently_scored']['min_hours_since_last_score']
            where_clauses.append(f"last_scored_at < NOW() - INTERVAL '{min_hours} hours'")

        # Build final query
        if where_clauses:
            query = f"{base_query} WHERE {' AND '.join(where_clauses)}"
        else:
            query = base_query

        # Add ordering
        ordering = self.config['strategies'][job.strategy]['ordering']
        if ordering == "oldest_first":
            query += " ORDER BY last_scored_at ASC"
        elif ordering == "newest_first":
            query += " ORDER BY last_scored_at DESC"
        elif ordering == "score_asc":
            query += " ORDER BY current_score ASC"
        elif ordering == "random":
            query += " ORDER BY RANDOM()"

        return query

    def _process_batch(self, batch: List[Dict], job: RescoreJob, dry_run: bool = False):
        """Process a batch of items"""
        self.logger.info(f"Processing batch of {len(batch)} items")

        start_time = time.time()
        results = []

        for item in batch:
            try:
                # Extract data
                data_id = item['id']
                data = item['data']
                data_type = item['data_type']
                metadata = item.get('metadata', {})
                old_score = item.get('current_score', 0.0)

                # Score the data
                score = self.classifier.classify(data, data_type, metadata)

                # Compare to old score
                score_change = score.overall_score - old_score

                if abs(score_change) > 0.10:
                    self.logger.info(
                        f"Significant score change for {data_id}: "
                        f"{old_score:.3f} -> {score.overall_score:.3f}"
                    )

                # Record metrics
                ITEMS_PROCESSED.labels(schedule=job.schedule_name, action=score.action).inc()
                SCORE_DISTRIBUTION.labels(schedule=job.schedule_name).observe(score.overall_score)

                results.append({
                    'id': data_id,
                    'score': score,
                    'old_score': old_score,
                    'score_change': score_change
                })

                self.processed_count += 1

            except Exception as e:
                self.logger.error(f"Error processing item {item.get('id')}: {e}")
                ITEMS_FAILED.labels(schedule=job.schedule_name).inc()
                self.failed_count += 1

        # Store results
        if not dry_run and results:
            self._store_results(results)

        # Record processing time
        duration = time.time() - start_time
        PROCESSING_TIME.labels(schedule=job.schedule_name).observe(duration)

        self.logger.debug(f"Batch processed in {duration:.2f}s")

    def _store_results(self, results: List[Dict]):
        """Store rescoring results to database"""
        try:
            with self.db_conn.cursor() as cur:
                for result in results:
                    score = result['score']

                    cur.execute("""
                        UPDATE dqs.quality_scores
                        SET
                            current_score = %s,
                            dimension_scores = %s,
                            category = %s,
                            action = %s,
                            labels = %s,
                            last_scored_at = %s,
                            rescore_attempts = rescore_attempts + 1,
                            previous_score = %s,
                            score_change = %s,
                            classifier_version = %s
                        WHERE id = %s
                    """, (
                        score.overall_score,
                        json.dumps(score.dimension_scores),
                        score.category,
                        score.action,
                        score.labels,
                        score.timestamp,
                        result['old_score'],
                        result['score_change'],
                        score.metadata.get('classifier_version'),
                        result['id']
                    ))

                self.db_conn.commit()

        except Exception as e:
            self.logger.error(f"Error storing results: {e}")
            self.db_conn.rollback()

    def _send_notification(self, message: str, channels: List[str]):
        """Send notification through configured channels"""
        self.logger.info(f"Notification: {message}")

        for channel in channels:
            try:
                if channel == "slack":
                    self._send_slack(message)
                elif channel == "email":
                    self._send_email(message)
                elif channel == "pagerduty":
                    self._send_pagerduty(message)
            except Exception as e:
                self.logger.error(f"Error sending notification to {channel}: {e}")

    def _send_slack(self, message: str):
        """Send Slack notification"""
        # Implement Slack webhook
        pass

    def _send_email(self, message: str):
        """Send email notification"""
        # Implement email sending
        pass

    def _send_pagerduty(self, message: str):
        """Send PagerDuty alert"""
        # Implement PagerDuty API call
        pass

    def list_schedules(self):
        """List all configured schedules"""
        print("Configured Schedules:")
        print("-" * 80)

        for name, config in self.config['schedules'].items():
            status = "✓ Enabled" if config['enabled'] else "✗ Disabled"
            print(f"\n{name} ({status})")
            print(f"  Schedule: {config['cron_human']}")
            print(f"  Strategy: {config['strategy']}")
            print(f"  Priority: {config['priority']}")
            print(f"  Timeout: {config['timeout_hours']}h")

    def status(self):
        """Show current status"""
        print("Rescore Orchestrator Status:")
        print("-" * 80)
        print(f"Running: {self.running}")
        print(f"Current Job: {self.current_job or 'None'}")
        print(f"Processed: {self.processed_count}")
        print(f"Failed: {self.failed_count}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='DQS Rescore Orchestrator')
    parser.add_argument('command', choices=['run', 'list', 'status'],
                       help='Command to execute')
    parser.add_argument('--schedule', '-s', help='Schedule name to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode (no changes)')
    parser.add_argument('--config', '-c', default='/etc/dqs/rescore_cron.json',
                       help='Path to rescore config')
    parser.add_argument('--classifier-config', default='/etc/dqs/classifier.json',
                       help='Path to classifier config')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = RescoreOrchestrator(
        config_path=args.config,
        classifier_config=args.classifier_config
    )

    # Start metrics server
    start_http_server(9146)

    # Execute command
    if args.command == 'list':
        orchestrator.list_schedules()

    elif args.command == 'status':
        orchestrator.status()

    elif args.command == 'run':
        if not args.schedule:
            print("Error: --schedule required for run command")
            sys.exit(1)

        try:
            orchestrator.execute_schedule(args.schedule, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
