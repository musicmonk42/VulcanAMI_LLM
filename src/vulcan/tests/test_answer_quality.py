"""
test_answer_quality.py - Tests for answer quality assessment functionality
Part of the VULCAN-AGI system test suite

This module tests the quality assessment functionality added to fix the
critical issue where the curiosity engine was blind to answer quality,
measuring only completion (did the query return something) rather than
quality (was the answer actually useful).
"""

import pytest
import tempfile
import os
from pathlib import Path


# Import the quality assessment function and related utilities
from vulcan.curiosity_engine.outcome_bridge import (
    assess_answer_quality,
    record_query_outcome,
    get_recent_outcomes,
    get_outcome_statistics,
    analyze_outcomes_for_gaps,
    reset_database,
    ANSWER_QUALITY_FAILURE_PATTERNS,
    RAW_DATA_DUMP_PATTERNS,
    MISCLASSIFICATION_PATTERNS,
)


class TestAssessAnswerQuality:
    """Test the assess_answer_quality function."""

    def test_good_answer(self):
        """Test that a normal helpful answer is marked as good."""
        response = "The answer to your question is 42. This is because..."
        quality = assess_answer_quality(response)
        assert quality == "good"

    def test_empty_response(self):
        """Test that empty or very short responses are marked as failed."""
        assert assess_answer_quality("") == "failed"
        assert assess_answer_quality("ok") == "failed"
        assert assess_answer_quality("yes") == "failed"

    def test_none_response(self):
        """Test that None response returns unknown."""
        assert assess_answer_quality(None) == "unknown"

    def test_failure_pattern_probability(self):
        """Test detection of probability engine rejection."""
        response = "This query does not involve probability concepts and cannot be processed."
        quality = assess_answer_quality(response)
        assert quality == "failed"

    def test_failure_pattern_not_applicable(self):
        """Test detection of 'not applicable' pattern."""
        response = "This analysis is not applicable to the given query type."
        quality = assess_answer_quality(response)
        assert quality == "failed"

    def test_failure_pattern_unable_to(self):
        """Test detection of 'unable to' pattern."""
        response = "I am unable to process this request due to constraints."
        quality = assess_answer_quality(response)
        assert quality == "failed"

    def test_raw_data_dump_single_pattern(self):
        """Test that a single raw data pattern doesn't trigger failure."""
        # Single pattern should not be marked as failed
        response = "The answer has confidence: 0.95 which indicates high certainty."
        quality = assess_answer_quality(response)
        # This might be "good" or "partial" depending on length
        assert quality in ("good", "partial")

    def test_raw_data_dump_multiple_patterns(self):
        """Test detection of raw data dumps with multiple patterns."""
        response = "proven: false, confidence: 0.2, method: symbolic_v2"
        quality = assess_answer_quality(response)
        assert quality == "failed"

    def test_misclassification_introspection(self):
        """Test detection of misclassified introspective queries."""
        # Query about writing (not about the AI itself)
        response = "I recognize this as an introspective query about my own nature."
        query = "Write me a poem about the ocean"
        quality = assess_answer_quality(response, query_text=query)
        assert quality == "failed"

    def test_misclassification_true_introspection(self):
        """Test that true introspection queries are not flagged as misclassified."""
        response = "I recognize this as an introspective query and will reflect on my capabilities."
        query = "Tell me about yourself and your capabilities"
        quality = assess_answer_quality(response, query_text=query)
        # Should NOT be marked as failed because query is actually about the AI
        assert quality in ("good", "partial")

    def test_low_confidence_partial(self):
        """Test that low confidence responses are marked as partial."""
        response = "Based on the limited information, the answer might be approximately 42, but I'm not certain."
        quality = assess_answer_quality(response, confidence=0.2)
        assert quality == "partial"

    def test_high_confidence_good(self):
        """Test that high confidence responses with sufficient content are good."""
        response = "The definitive answer to your mathematical question is 42. This is calculated by..."
        quality = assess_answer_quality(response, confidence=0.9)
        assert quality == "good"

    def test_short_response_partial(self):
        """Test that short but valid responses are marked as partial."""
        response = "The answer is 42."  # Under 50 chars
        quality = assess_answer_quality(response)
        assert quality == "partial"

    def test_all_failure_patterns_detected(self):
        """Test that all defined failure patterns are detected."""
        for pattern in ANSWER_QUALITY_FAILURE_PATTERNS:
            response = f"The system says: {pattern} for this query."
            quality = assess_answer_quality(response)
            assert quality == "failed", f"Pattern '{pattern}' was not detected"

    def test_introspection_with_various_query_patterns(self):
        """Test that various introspective query patterns are detected correctly."""
        response = "I recognize this as an introspective query about my capabilities."
        
        # These queries should NOT be flagged as misclassification
        introspective_queries = [
            "Tell me about yourself",
            "What are you?",
            "Who are you?",
            "Describe your capabilities",
            "Tell me about you",
        ]
        
        for query in introspective_queries:
            quality = assess_answer_quality(response, query_text=query)
            assert quality in ("good", "partial"), f"Query '{query}' was incorrectly marked as failed"
        
        # These queries SHOULD be flagged as misclassification
        non_introspective_queries = [
            "Write me a poem",
            "Compute 2+2",
            "What is the capital of France?",
        ]
        
        for query in non_introspective_queries:
            quality = assess_answer_quality(response, query_text=query)
            assert quality == "failed", f"Non-introspective query '{query}' was not flagged as misclassified"


class TestRecordQueryOutcomeWithQuality:
    """Test recording query outcomes with quality metrics."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_outcomes.db"
            yield db_path

    def test_record_with_response_text(self, temp_db):
        """Test recording outcome with response text for auto quality assessment."""
        success = record_query_outcome(
            query_id="test_q1",
            status="success",
            routing_time_ms=100.0,
            total_time_ms=500.0,
            complexity=0.5,
            query_type="reasoning",
            response_text="The answer is 42. This is because the question asked for the meaning of life.",
            db_path=temp_db,
        )
        assert success is True

        # Verify the outcome was recorded
        outcomes = get_recent_outcomes(minutes=60, db_path=temp_db)
        assert len(outcomes) == 1
        assert outcomes[0]["answer_quality"] == "good"

    def test_record_with_failed_quality(self, temp_db):
        """Test recording outcome that should be marked as failed quality."""
        success = record_query_outcome(
            query_id="test_q2",
            status="success",
            routing_time_ms=100.0,
            total_time_ms=500.0,
            complexity=0.5,
            query_type="reasoning",
            response_text="This query does not involve probability concepts.",
            db_path=temp_db,
        )
        assert success is True

        outcomes = get_recent_outcomes(minutes=60, db_path=temp_db)
        assert len(outcomes) == 1
        assert outcomes[0]["answer_quality"] == "failed"

    def test_record_with_explicit_quality(self, temp_db):
        """Test recording outcome with explicitly provided quality."""
        success = record_query_outcome(
            query_id="test_q3",
            status="success",
            routing_time_ms=100.0,
            total_time_ms=500.0,
            complexity=0.5,
            query_type="reasoning",
            response_text="Some response",
            answer_quality="good",  # Explicitly override
            db_path=temp_db,
        )
        assert success is True

        outcomes = get_recent_outcomes(minutes=60, db_path=temp_db)
        assert len(outcomes) == 1
        assert outcomes[0]["answer_quality"] == "good"

    def test_record_without_quality(self, temp_db):
        """Test recording outcome without quality data."""
        success = record_query_outcome(
            query_id="test_q4",
            status="success",
            routing_time_ms=100.0,
            total_time_ms=500.0,
            complexity=0.5,
            query_type="reasoning",
            db_path=temp_db,
        )
        assert success is True

        outcomes = get_recent_outcomes(minutes=60, db_path=temp_db)
        assert len(outcomes) == 1
        assert outcomes[0]["answer_quality"] is None


class TestQualityStatistics:
    """Test quality statistics computation."""

    @pytest.fixture
    def temp_db_with_data(self):
        """Create a temporary database with test data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_outcomes.db"
            
            # Record some outcomes with mixed quality
            record_query_outcome(
                query_id="q1", status="success", routing_time_ms=100.0,
                total_time_ms=500.0, complexity=0.5, query_type="reasoning",
                response_text="The answer is 42. This is a comprehensive explanation.",
                db_path=db_path,
            )
            record_query_outcome(
                query_id="q2", status="success", routing_time_ms=100.0,
                total_time_ms=500.0, complexity=0.5, query_type="reasoning",
                response_text="does not involve probability concepts",
                db_path=db_path,
            )
            record_query_outcome(
                query_id="q3", status="success", routing_time_ms=100.0,
                total_time_ms=500.0, complexity=0.5, query_type="reasoning",
                response_text="Another good answer with sufficient detail and explanation.",
                db_path=db_path,
            )
            record_query_outcome(
                query_id="q4", status="success", routing_time_ms=100.0,
                total_time_ms=500.0, complexity=0.5, query_type="reasoning",
                response_text="proven: false, confidence: 0.1",
                db_path=db_path,
            )
            
            yield db_path

    def test_quality_statistics(self, temp_db_with_data):
        """Test that quality statistics are computed correctly."""
        stats = get_outcome_statistics(db_path=temp_db_with_data)
        
        assert stats.total == 4
        assert stats.quality_good_count == 2  # q1 and q3
        assert stats.quality_failed_count == 2  # q2 and q4
        assert stats.quality_success_rate == pytest.approx(0.5, rel=0.01)


class TestQualityGapDetection:
    """Test gap detection based on quality metrics."""

    def test_detect_low_quality_gap(self):
        """Test that low quality rate is detected as a gap."""
        outcomes = [
            {"query_id": f"q{i}", "answer_quality": "failed"}
            for i in range(8)
        ] + [
            {"query_id": f"qg{i}", "answer_quality": "good"}
            for i in range(2)
        ]
        
        gaps = analyze_outcomes_for_gaps(outcomes)
        
        # Should detect low answer quality gap
        quality_gaps = [g for g in gaps if "quality" in g["gap_type"]]
        assert len(quality_gaps) > 0

    def test_detect_answer_failure_patterns_gap(self):
        """Test detection of high answer failure rate as a gap."""
        outcomes = [
            {"query_id": f"q{i}", "answer_quality": "failed"}
            for i in range(5)
        ]
        
        gaps = analyze_outcomes_for_gaps(outcomes)
        
        # Should detect answer failure patterns gap
        pattern_gaps = [g for g in gaps if "failure" in g["gap_type"]]
        assert len(pattern_gaps) > 0

    def test_no_gap_for_high_quality(self):
        """Test that no quality gaps are detected when quality is high."""
        outcomes = [
            {"query_id": f"q{i}", "answer_quality": "good"}
            for i in range(10)
        ]
        
        gaps = analyze_outcomes_for_gaps(outcomes)
        
        # Should NOT detect quality-related gaps
        quality_gaps = [g for g in gaps if "quality" in g["gap_type"] or "failure_patterns" in g["gap_type"]]
        assert len(quality_gaps) == 0

    def test_no_gap_without_quality_data(self):
        """Test that no quality gaps are detected when no quality data exists."""
        outcomes = [
            {"query_id": f"q{i}", "status": "success", "routing_time_ms": 100.0}
            for i in range(10)
        ]
        
        gaps = analyze_outcomes_for_gaps(outcomes)
        
        # Should NOT detect quality-related gaps (no quality data)
        quality_gaps = [g for g in gaps if "quality" in g["gap_type"]]
        assert len(quality_gaps) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
