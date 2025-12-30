#!/usr/bin/env python3
"""
VULCAN Distillation Monitoring Script.

Shows capture statistics, storage status, and training progress.
Run with --continuous for real-time monitoring.

Usage:
    python scripts/monitor_distillation.py              # One-time status
    python scripts/monitor_distillation.py --continuous # Live monitoring
    python scripts/monitor_distillation.py --interval 30 --continuous  # Custom refresh
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Add src to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src = os.path.join(_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs during monitoring
    format="%(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_bytes(bytes_val: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} TB"


def format_time_ago(timestamp: Optional[float]) -> str:
    """Format timestamp as human-readable time ago."""
    if timestamp is None:
        return "never"
    
    delta = time.time() - timestamp
    if delta < 60:
        return f"{int(delta)}s ago"
    elif delta < 3600:
        return f"{int(delta / 60)}m ago"
    elif delta < 86400:
        return f"{int(delta / 3600)}h ago"
    else:
        return f"{int(delta / 86400)}d ago"


def get_distiller_status() -> Optional[Dict[str, Any]]:
    """Get status from the global distiller instance."""
    try:
        from vulcan.distillation import get_knowledge_distiller
        distiller = get_knowledge_distiller()
        if distiller:
            return distiller.get_status()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error getting distiller status: {e}")
    return None


def get_storage_stats(storage_dir: str = "data/distillation") -> Optional[Dict[str, Any]]:
    """Get statistics from distillation storage."""
    try:
        from vulcan.distillation.storage import DistillationStorageBackend
        storage = DistillationStorageBackend(storage_path=storage_dir)
        return storage.get_stats()
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Error getting storage stats: {e}")
    return None


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def show_status(storage_dir: str = "data/distillation") -> None:
    """Display comprehensive distillation status."""
    width = 70
    
    print("\n" + "=" * width)
    print("VULCAN DISTILLATION SYSTEM STATUS".center(width))
    print("=" * width)
    
    # Timestamp
    print(f"\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Distiller status
    distiller_status = get_distiller_status()
    if distiller_status:
        stats = distiller_status.get('stats', {})
        state = distiller_status.get('state', {})
        config = distiller_status.get('config', {})
        
        print("\n" + "-" * width)
        print("📊 CAPTURE STATISTICS")
        print("-" * width)
        
        captured = stats.get('examples_captured', 0)
        rejected = stats.get('examples_rejected', 0)
        total = captured + rejected
        
        print(f"  Examples captured:  {captured:,}")
        print(f"  Examples rejected:  {rejected:,}")
        
        if total > 0:
            rate = captured / total * 100
            print(f"  Capture rate:       {rate:.1f}%")
        
        print("\n" + "-" * width)
        print("🔒 PRIVACY & SECURITY")
        print("-" * width)
        print(f"  PII redactions:     {stats.get('pii_redactions', 0):,}")
        print(f"  Secrets detected:   {stats.get('secrets_detected', 0):,}")
        print(f"  Governance blocks:  {stats.get('governance_sensitive_rejections', 0):,}")
        
        print("\n" + "-" * width)
        print("✨ QUALITY")
        print("-" * width)
        avg_quality = stats.get('average_quality_score', 0.0)
        print(f"  Average score:      {avg_quality:.3f}")
        
        # Rejection breakdown
        rejection_reasons = stats.get('rejection_reasons', {})
        if rejection_reasons:
            print("\n  ⚠️  Rejection Breakdown:")
            sorted_reasons = sorted(
                rejection_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for reason, count in sorted_reasons[:5]:
                print(f"    {reason:20s} {count:,}")
        
        print("\n" + "-" * width)
        print("💾 BUFFER STATUS")
        print("-" * width)
        buffer_size = state.get('buffer_size', 0)
        max_buffer = config.get('max_buffer_size', 100)
        print(f"  Current size:       {buffer_size:,}")
        print(f"  Max size:           {max_buffer:,}")
        if max_buffer > 0:
            fill_pct = (buffer_size / max_buffer) * 100
            print(f"  Fill percentage:    {fill_pct:.1f}%")
    else:
        print("\n⚠️  Distiller not initialized")
        print("   Run your application to initialize the distiller.")
    
    # Storage status
    print("\n" + "-" * width)
    print("💿 STORAGE")
    print("-" * width)
    
    storage_stats = get_storage_stats(storage_dir)
    if storage_stats:
        print(f"  Total examples:     {storage_stats.get('total_examples', 0):,}")
        print(f"  File size:          {format_bytes(storage_stats.get('file_size_bytes', 0))}")
        
        encryption = storage_stats.get('encryption_enabled', False)
        print(f"  Encryption:         {'enabled' if encryption else 'disabled'}")
        
        created_at = storage_stats.get('created_at')
        if created_at:
            created = datetime.fromtimestamp(created_at)
            print(f"  Created:            {created.strftime('%Y-%m-%d %H:%M:%S')}")
        
        last_write = storage_stats.get('last_write')
        if last_write:
            print(f"  Last write:         {format_time_ago(last_write)}")
    else:
        # Check if storage directory exists
        full_path = os.path.join(_root, storage_dir)
        if os.path.exists(full_path):
            print(f"  Storage directory:  {storage_dir} (exists)")
            # Check for examples file
            examples_file = os.path.join(full_path, "examples.jsonl")
            if os.path.exists(examples_file):
                size = os.path.getsize(examples_file)
                print(f"  Examples file:      {format_bytes(size)}")
                
                # Count lines
                try:
                    with open(examples_file, 'r') as f:
                        line_count = sum(1 for _ in f)
                    print(f"  Total examples:     {line_count:,}")
                except Exception:
                    pass
            else:
                print("  Examples file:      not created yet")
        else:
            print(f"  Storage directory:  {storage_dir} (not created)")
    
    # Training hints
    print("\n" + "-" * width)
    print("📘 TRAINING")
    print("-" * width)
    print("  To train from captured examples:")
    print("    python src/training/train_llm_with_self_improvement.py \\")
    print(f"        --distillation-storage {storage_dir}")
    
    print("\n" + "=" * width)


def continuous_monitor(interval: int = 60, storage_dir: str = "data/distillation") -> None:
    """Continuously monitor with refresh interval."""
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            show_status(storage_dir)
            
            print(f"\n🔄 Refreshing in {interval}s... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Monitor VULCAN distillation system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-time status check
  python scripts/monitor_distillation.py
  
  # Continuous monitoring with default 60s refresh
  python scripts/monitor_distillation.py --continuous
  
  # Custom refresh interval
  python scripts/monitor_distillation.py --continuous --interval 30
  
  # Custom storage directory
  python scripts/monitor_distillation.py --storage-dir /path/to/storage
        """
    )
    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Continuously monitor with refresh"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=60,
        help="Refresh interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--storage-dir", "-s",
        type=str,
        default="data/distillation",
        help="Path to distillation storage directory"
    )
    
    args = parser.parse_args()
    
    if args.continuous:
        continuous_monitor(args.interval, args.storage_dir)
    else:
        show_status(args.storage_dir)


if __name__ == "__main__":
    main()
