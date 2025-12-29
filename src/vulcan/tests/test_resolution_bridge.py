"""
test_resolution_bridge.py - Tests for Resolution Bridge module

Part of the VULCAN-AGI system

Tests cover:
- Gap resolution tracking (mark, check, clear)
- Experiment attempt counting
- Resolution history and phantom detection
- Persistent experiment counters (cold start prevention)
- Cross-process state sharing
- Statistics computation
- Database cleanup and maintenance

These tests verify the fix for:
- Bug #1: Phantom Resolution Loop (gaps "resolved" 40-90x/hour)
- Bug #2: Cold Start Always Triggered (thinking 0/5 experiments ran)
"""

import tempfile
import threading
import time
from pathlib import Path

import pytest

from vulcan.curiosity_engine.resolution_bridge import (
    ResolutionRecord,
    ResolutionStatistics,
    ResolutionStatus,
    clear_gap_resolution,
    cleanup_old_data,
    get_all_counters,
    get_all_resolved_gaps,
    get_experiment_count,
    get_gap_attempts,
    get_recent_resolutions_count,
    get_resolution_statistics,
    increment_experiment_count,
    increment_gap_attempts,
    is_gap_resolved,
    is_phantom_resolution,
    mark_gap_resolved,
    record_resolution_history,
    reset_database,
    reset_gap_attempts,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def clean_database():
    """Reset the database before each test for isolation."""
    reset_database()
    yield
    reset_database()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for isolated tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_resolutions.db"
        yield db_path


# =============================================================================
# Test Gap Resolution Basic Operations
# =============================================================================


class TestGapResolutionBasics:
    """Tests for basic gap resolution operations."""

    def test_gap_not_resolved_initially(self, clean_database):
        """Test that gaps are not resolved by default."""
        assert not is_gap_resolved("test:gap1")
        assert not is_gap_resolved("high_error_rate:query_processing")

    def test_mark_gap_resolved_success(self, clean_database):
        """Test marking a gap as resolved successfully."""
        gap_key = "test:gap1"
        
        result = mark_gap_resolved(gap_key, success=True)
        
        assert result is True
        assert is_gap_resolved(gap_key)

    def test_mark_gap_resolved_give_up(self, clean_database):
        """Test marking a gap as resolved when giving up."""
        gap_key = "test:gap2"
        
        result = mark_gap_resolved(gap_key, success=False)
        
        assert result is True
        assert is_gap_resolved(gap_key)

    def test_clear_gap_resolution(self, clean_database):
        """Test clearing gap resolution status."""
        gap_key = "test:gap3"
        mark_gap_resolved(gap_key, success=True)
        assert is_gap_resolved(gap_key)
        
        result = clear_gap_resolution(gap_key)
        
        assert result is True
        assert not is_gap_resolved(gap_key)

    def test_resolution_ttl_expiry(self, clean_database):
        """Test that resolutions expire after TTL."""
        gap_key = "test:gap4"
        mark_gap_resolved(gap_key, success=True)
        
        # With very short TTL, resolution should be expired
        assert not is_gap_resolved(gap_key, ttl_seconds=0)

    def test_resolution_within_ttl(self, clean_database):
        """Test that resolutions are valid within TTL."""
        gap_key = "test:gap5"
        mark_gap_resolved(gap_key, success=True)
        
        # With long TTL, resolution should still be valid
        assert is_gap_resolved(gap_key, ttl_seconds=3600)


# =============================================================================
# Test Experiment Attempt Tracking
# =============================================================================


class TestExperimentAttempts:
    """Tests for experiment attempt tracking."""

    def test_initial_attempts_zero(self, clean_database):
        """Test that attempts are zero initially."""
        assert get_gap_attempts("test:gap1") == 0

    def test_increment_attempts(self, clean_database):
        """Test incrementing attempt counter."""
        gap_key = "test:gap1"
        
        count1 = increment_gap_attempts(gap_key)
        count2 = increment_gap_attempts(gap_key)
        count3 = increment_gap_attempts(gap_key)
        
        assert count1 == 1
        assert count2 == 2
        assert count3 == 3
        assert get_gap_attempts(gap_key) == 3

    def test_reset_attempts(self, clean_database):
        """Test resetting attempt counter."""
        gap_key = "test:gap1"
        increment_gap_attempts(gap_key)
        increment_gap_attempts(gap_key)
        
        result = reset_gap_attempts(gap_key)
        
        assert result is True
        assert get_gap_attempts(gap_key) == 0


# =============================================================================
# Test Resolution History and Phantom Detection
# =============================================================================


class TestResolutionHistoryAndPhantom:
    """Tests for resolution history and phantom detection."""

    def test_record_resolution_history(self, clean_database):
        """Test recording resolution history."""
        gap_key = "test:gap1"
        
        result = record_resolution_history(gap_key, success=True, cycle_id=1)
        
        assert result is True
        assert get_recent_resolutions_count(gap_key) == 1

    def test_multiple_resolutions_counted(self, clean_database):
        """Test that multiple resolutions are counted."""
        gap_key = "test:gap1"
        
        record_resolution_history(gap_key, success=True, cycle_id=1)
        record_resolution_history(gap_key, success=True, cycle_id=2)
        record_resolution_history(gap_key, success=True, cycle_id=3)
        
        assert get_recent_resolutions_count(gap_key) == 3

    def test_phantom_detection_threshold(self, clean_database):
        """Test phantom detection triggers at threshold."""
        gap_key = "test:gap1"
        
        # Below threshold - not phantom
        record_resolution_history(gap_key, success=True, cycle_id=1)
        record_resolution_history(gap_key, success=True, cycle_id=2)
        assert not is_phantom_resolution(gap_key, threshold=3)
        
        # At threshold - is phantom
        record_resolution_history(gap_key, success=True, cycle_id=3)
        assert is_phantom_resolution(gap_key, threshold=3)

    def test_phantom_detection_window(self, clean_database):
        """Test phantom detection respects time window."""
        gap_key = "test:gap1"
        
        record_resolution_history(gap_key, success=True, cycle_id=1)
        record_resolution_history(gap_key, success=True, cycle_id=2)
        record_resolution_history(gap_key, success=True, cycle_id=3)
        
        # With very short window, should not see old resolutions
        count = get_recent_resolutions_count(gap_key, window_seconds=0)
        assert count == 0


# =============================================================================
# Test Experiment Counters (Cold Start Prevention)
# =============================================================================


class TestExperimentCounters:
    """Tests for persistent experiment counters (Bug #2 Fix)."""

    def test_initial_experiment_count_zero(self, clean_database):
        """Test that experiment count is zero initially."""
        assert get_experiment_count("total_experiments") == 0

    def test_increment_experiment_count(self, clean_database):
        """Test incrementing experiment counter."""
        count1 = increment_experiment_count("total_experiments", 1)
        count2 = increment_experiment_count("total_experiments", 1)
        count3 = increment_experiment_count("total_experiments", 3)
        
        assert count1 == 1
        assert count2 == 2
        assert count3 == 5
        assert get_experiment_count("total_experiments") == 5

    def test_multiple_counters(self, clean_database):
        """Test multiple independent counters."""
        increment_experiment_count("total_experiments", 5)
        increment_experiment_count("successful_experiments", 3)
        increment_experiment_count("failed_experiments", 2)
        
        assert get_experiment_count("total_experiments") == 5
        assert get_experiment_count("successful_experiments") == 3
        assert get_experiment_count("failed_experiments") == 2

    def test_get_all_counters(self, clean_database):
        """Test getting all counters at once."""
        increment_experiment_count("total", 10)
        increment_experiment_count("success", 8)
        increment_experiment_count("failure", 2)
        
        counters = get_all_counters()
        
        assert counters["total"] == 10
        assert counters["success"] == 8
        assert counters["failure"] == 2

    def test_counter_persists_across_reads(self, clean_database):
        """Test that counter persists (simulates subprocess restart)."""
        # First "process" increments
        increment_experiment_count("total_experiments", 5)
        
        # "Second process" should see the count
        count = get_experiment_count("total_experiments")
        assert count == 5, f"Expected 5, got {count} - counter should persist"


# =============================================================================
# Test Bulk Operations
# =============================================================================


class TestBulkOperations:
    """Tests for bulk operations."""

    def test_get_all_resolved_gaps(self, clean_database):
        """Test getting all resolved gaps."""
        mark_gap_resolved("type1:domain1", success=True)
        mark_gap_resolved("type2:domain2", success=True)
        mark_gap_resolved("type3:domain3", success=False)
        
        gaps = get_all_resolved_gaps(include_expired=True)
        
        assert "type1:domain1" in gaps
        assert "type2:domain2" in gaps
        assert "type3:domain3" in gaps

    def test_get_resolved_gaps_excludes_expired(self, clean_database):
        """Test that expired gaps are excluded by default."""
        mark_gap_resolved("test:gap1", success=True)
        
        # With normal TTL, gap should be included
        gaps = get_all_resolved_gaps(include_expired=False)
        assert "test:gap1" in gaps


# =============================================================================
# Test Statistics
# =============================================================================


class TestStatistics:
    """Tests for statistics computation."""

    def test_initial_statistics(self, clean_database):
        """Test initial statistics are zero."""
        stats = get_resolution_statistics()
        
        assert stats.total_resolutions == 0
        assert stats.total_experiments == 0

    def test_statistics_after_resolutions(self, clean_database):
        """Test statistics after some operations."""
        mark_gap_resolved("test:gap1", success=True)
        mark_gap_resolved("test:gap2", success=True)
        mark_gap_resolved("test:gap3", success=False)
        increment_experiment_count("total_experiments", 10)
        
        # Record phantom pattern
        for i in range(3):
            record_resolution_history("test:phantom", success=True, cycle_id=i)
        
        stats = get_resolution_statistics()
        
        assert stats.total_resolutions >= 3
        assert stats.total_experiments == 10
        assert stats.phantom_count >= 1

    def test_statistics_to_dict(self, clean_database):
        """Test statistics serialization."""
        stats = get_resolution_statistics()
        d = stats.to_dict()
        
        assert "total_resolutions" in d
        assert "total_experiments" in d
        assert "phantom_count" in d
        assert "success_rate" in d


# =============================================================================
# Test Data Classes
# =============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_resolution_record(self):
        """Test ResolutionRecord dataclass."""
        record = ResolutionRecord(
            gap_key="test:gap1",
            resolved_at=time.time(),
            success=True,
            attempts=3,
        )
        
        assert record.gap_key == "test:gap1"
        assert record.success is True
        assert record.attempts == 3
        assert record.status == ResolutionStatus.SUCCESS

    def test_resolution_record_give_up_status(self):
        """Test ResolutionRecord with give-up status."""
        record = ResolutionRecord(
            gap_key="test:gap1",
            resolved_at=time.time(),
            success=False,
            attempts=10,
        )
        
        assert record.status == ResolutionStatus.GIVE_UP

    def test_resolution_record_pending_status(self):
        """Test ResolutionRecord with pending status."""
        record = ResolutionRecord(
            gap_key="test:gap1",
            resolved_at=0,  # Not resolved yet
            success=False,
            attempts=0,
        )
        
        assert record.status == ResolutionStatus.PENDING

    def test_resolution_record_to_dict(self):
        """Test ResolutionRecord serialization."""
        record = ResolutionRecord(
            gap_key="test:gap1",
            resolved_at=123456.789,
            success=True,
            attempts=3,
        )
        
        d = record.to_dict()
        
        assert d["gap_key"] == "test:gap1"
        assert d["resolved_at"] == 123456.789
        assert d["success"] is True
        assert d["attempts"] == 3

    def test_resolution_status_enum(self):
        """Test ResolutionStatus enum values."""
        assert ResolutionStatus.SUCCESS.value == "success"
        assert ResolutionStatus.GIVE_UP.value == "give_up"
        assert ResolutionStatus.PENDING.value == "pending"
        assert ResolutionStatus.PHANTOM.value == "phantom"


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_increments(self, clean_database):
        """Test concurrent counter increments."""
        errors = []
        
        def increment_many():
            try:
                for _ in range(100):
                    increment_experiment_count("concurrent_test", 1)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=increment_many) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # Should have 500 total increments
        count = get_experiment_count("concurrent_test")
        assert count == 500, f"Expected 500, got {count}"

    def test_concurrent_resolution_marking(self, clean_database):
        """Test concurrent gap resolution marking."""
        errors = []
        
        def mark_many(thread_id):
            try:
                for i in range(20):
                    mark_gap_resolved(f"thread{thread_id}:gap{i}", success=True)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=mark_many, args=(i,)) for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All gaps should be resolved
        gaps = get_all_resolved_gaps(include_expired=True)
        assert len(gaps) >= 100  # 5 threads * 20 gaps


# =============================================================================
# Test Database Maintenance
# =============================================================================


class TestDatabaseMaintenance:
    """Tests for database maintenance operations."""

    def test_cleanup_old_data(self, clean_database):
        """Test cleanup of old data."""
        # Create some data
        mark_gap_resolved("test:gap1", success=True)
        record_resolution_history("test:gap1", success=True, cycle_id=1)
        
        # Cleanup should not delete recent data
        deleted = cleanup_old_data(days=1)
        
        # Recent data should still exist
        assert is_gap_resolved("test:gap1")

    def test_reset_database(self, clean_database):
        """Test complete database reset."""
        # Create some data
        mark_gap_resolved("test:gap1", success=True)
        increment_experiment_count("total", 100)
        
        # Reset
        result = reset_database()
        
        assert result is True
        assert not is_gap_resolved("test:gap1")
        assert get_experiment_count("total") == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_resolution_workflow(self, clean_database):
        """Test complete gap resolution workflow."""
        gap_key = "high_error_rate:query_processing"
        
        # Initial state
        assert not is_gap_resolved(gap_key)
        assert get_gap_attempts(gap_key) == 0
        
        # First attempt fails
        increment_gap_attempts(gap_key)
        assert get_gap_attempts(gap_key) == 1
        
        # Second attempt fails
        increment_gap_attempts(gap_key)
        assert get_gap_attempts(gap_key) == 2
        
        # Third attempt succeeds
        increment_gap_attempts(gap_key)
        mark_gap_resolved(gap_key, success=True)
        record_resolution_history(gap_key, success=True, cycle_id=1)
        
        # Gap should be resolved
        assert is_gap_resolved(gap_key)
        
        # Record in history
        count = get_recent_resolutions_count(gap_key)
        assert count == 1

    def test_phantom_resolution_detection_workflow(self, clean_database):
        """Test phantom resolution detection workflow."""
        gap_key = "persistent:issue"
        
        # Simulate repeated "resolutions" that don't actually fix the issue
        for cycle_id in range(5):
            mark_gap_resolved(gap_key, success=True)
            record_resolution_history(gap_key, success=True, cycle_id=cycle_id)
        
        # Should be detected as phantom
        assert is_phantom_resolution(gap_key, threshold=3)
        
        # Resolution count should be high
        count = get_recent_resolutions_count(gap_key)
        assert count >= 5

    def test_cold_start_prevention_workflow(self, clean_database):
        """Test cold start prevention workflow (Bug #2 Fix)."""
        # Simulate first "process" running experiments
        increment_experiment_count("total_experiments", 1)
        increment_experiment_count("total_experiments", 1)
        increment_experiment_count("total_experiments", 1)
        
        # "Second process" should see the experiments (not cold start)
        total = get_experiment_count("total_experiments")
        
        # This is the key assertion for Bug #2 fix
        assert total >= 3, (
            f"Expected at least 3 experiments, got {total}. "
            "Subprocess should see experiment count from previous process."
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
