#!/usr/bin/env python3
"""
VulcanAMI Data Quality System - Test Suite
Comprehensive testing and validation for DQS components
"""

import json
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, List

from dqs_classifier import DataQualityClassifier, QualityScore


class TestDataQualityClassifier(unittest.TestCase):
    """Test cases for Data Quality Classifier"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize classifier once for all tests"""
        cls.classifier = DataQualityClassifier()
    
    def test_perfect_data(self):
        """Test data with excellent quality"""
        data = {
            "title": "Machine Learning Research",
            "content": "This is a well-structured research document.",
            "author": "Research Team",
            "created_at": datetime.utcnow().isoformat(),
            "tags": ["ml", "research", "quality"]
        }
        
        score = self.classifier.classify(data, "json", {"source": "arxiv.org"})
        
        self.assertGreaterEqual(score.overall_score, 0.75)
        self.assertIn(score.category, ["good", "excellent"])
        self.assertEqual(score.action, "accept")
    
    def test_pii_detection(self):
        """Test PII detection accuracy"""
        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john.doe@example.com",
            "phone": "+1-555-123-4567",
            "content": "Contact information for John Doe."
        }
        
        score = self.classifier.classify(data, "json", {"source": "internal.com"})
        
        self.assertLess(score.dimension_scores.get("pii_confidence", 1.0), 0.90)
        self.assertIn("contains_pii", score.labels)
    
    def test_incomplete_data(self):
        """Test incomplete data scoring"""
        data = {
            "field1": None,
            "field2": "",
            "field3": "N/A",
            "field4": "actual data"
        }
        
        score = self.classifier.classify(data, "json", {"source": "test.com"})
        
        self.assertLess(score.dimension_scores.get("completeness_score", 1.0), 0.80)
        self.assertIn("incomplete_fields", score.labels)
    
    def test_stale_data(self):
        """Test data freshness scoring"""
        old_date = (datetime.utcnow() - timedelta(days=200)).isoformat()
        
        data = {
            "content": "Old data",
            "created_at": old_date,
            "updated_at": old_date
        }
        
        score = self.classifier.classify(data, "json", {"source": "test.com"})
        
        self.assertLess(score.dimension_scores.get("data_freshness", 1.0), 0.50)
        self.assertIn("stale_data", score.labels)
    
    def test_untrusted_source(self):
        """Test source credibility scoring"""
        data = {
            "content": "Test content"
        }
        
        score = self.classifier.classify(data, "json", {"source": "unknown-site.com"})
        
        self.assertLessEqual(score.dimension_scores.get("source_credibility", 1.0), 0.50)
    
    def test_syntax_errors(self):
        """Test syntactic validation"""
        # Malformed JSON string
        data = '{"key": "value", "missing_quote: "value"}'
        
        score = self.classifier.classify(data, "json", {"source": "test.com"})
        
        self.assertLess(score.dimension_scores.get("syntactic_completeness", 1.0), 0.80)
    
    def test_score_caching(self):
        """Test score caching functionality"""
        data = {"content": "Test data for caching"}
        metadata = {"source": "cache-test.com"}
        
        # First call - should not be cached
        start_time = time.time()
        score1 = self.classifier.classify(data, "json", metadata)
        first_duration = time.time() - start_time
        
        # Second call - should be cached
        start_time = time.time()
        score2 = self.classifier.classify(data, "json", metadata)
        second_duration = time.time() - start_time
        
        # Cached call should be faster
        self.assertLess(second_duration, first_duration)
        self.assertEqual(score1.overall_score, score2.overall_score)
    
    def test_multi_label_classification(self):
        """Test multi-label classification"""
        data = {
            "name": "Jane Doe",
            "ssn": "987-65-4321",
            "field1": None,
            "field2": "",
            "created_at": (datetime.utcnow() - timedelta(days=250)).isoformat()
        }
        
        score = self.classifier.classify(data, "json", {"source": "unknown.com"})
        
        # Should have multiple labels
        self.assertGreater(len(score.labels), 0)
        self.assertTrue(
            any(label in score.labels for label in ["contains_pii", "incomplete_fields", "stale_data"])
        )
    
    def test_category_thresholds(self):
        """Test category assignment based on score thresholds"""
        test_cases = [
            (0.95, "excellent", "accept"),
            (0.85, "good", "accept"),
            (0.70, "fair", "warn"),
            (0.50, "poor", "quarantine"),
            (0.25, "unacceptable", "reject")
        ]
        
        for target_score, expected_category, expected_action in test_cases:
            # Create mock score
            category, action = self.classifier._categorize_score(target_score)
            self.assertEqual(category, expected_category)
            self.assertEqual(action, expected_action)
    
    def test_dimension_weights(self):
        """Test dimension weight configuration"""
        total_weight = sum(
            dim['weight'] for dim in self.classifier.config['weights'].values()
            if dim['enabled']
        )
        
        # Total weight should be approximately 1.0
        self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestRescoreSchedules(unittest.TestCase):
    """Test cases for rescore scheduling"""
    
    def test_schedule_configuration(self):
        """Test schedule configuration loading"""
        with open('/etc/dqs/rescore_cron.json', 'r') as f:
            config = json.load(f)
        
        # Verify all schedules have required fields
        for schedule_name, schedule_config in config['schedules'].items():
            self.assertIn('cron', schedule_config)
            self.assertIn('strategy', schedule_config)
            self.assertIn('enabled', schedule_config)
            self.assertIn('priority', schedule_config)
    
    def test_strategy_configuration(self):
        """Test strategy configuration"""
        with open('/etc/dqs/rescore_cron.json', 'r') as f:
            config = json.load(f)
        
        # Verify all strategies have required fields
        for strategy_name, strategy_config in config['strategies'].items():
            self.assertIn('batch_size', strategy_config)
            self.assertIn('ordering', strategy_config)
            self.assertIn('checkpoint_interval', strategy_config)


class TestPerformance(unittest.TestCase):
    """Performance and load testing"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize classifier"""
        cls.classifier = DataQualityClassifier()
    
    def test_classification_speed(self):
        """Test classification performance"""
        data = {
            "content": "Test data for performance testing",
            "metadata": "Additional metadata"
        }
        
        # Time 100 classifications
        start_time = time.time()
        for _ in range(100):
            self.classifier.classify(data, "json", {"source": "test.com"})
        duration = time.time() - start_time
        
        # Should process at least 10 items per second
        items_per_second = 100 / duration
        self.assertGreater(items_per_second, 10)
        print(f"Classification speed: {items_per_second:.1f} items/second")
    
    def test_memory_usage(self):
        """Test memory usage during classification"""
        import os

        import psutil
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Classify 1000 items
        data = {"content": "Test data"}
        for _ in range(1000):
            self.classifier.classify(data, "json", {"source": "test.com"})
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500 MB)
        self.assertLess(memory_increase, 500)
        print(f"Memory increase: {memory_increase:.1f} MB")


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_database_connection(self):
        """Test database connectivity"""
        import psycopg2
        
        try:
            conn = psycopg2.connect(
                host='postgres',
                port=5432,
                database='vulcanami',
                user='dqs'
            )
            conn.close()
            success = True
        except Exception as e:
            success = False
            print(f"Database connection error: {e}")
        
        self.assertTrue(success)
    
    def test_redis_connection(self):
        """Test Redis connectivity"""
        import redis
        
        try:
            r = redis.Redis(host='redis', port=6379, db=1)
            r.ping()
            success = True
        except Exception as e:
            success = False
            print(f"Redis connection error: {e}")
        
        self.assertTrue(success)
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from classification to storage"""
        classifier = DataQualityClassifier()
        
        # Create test data
        data = {
            "title": "Integration Test",
            "content": "End-to-end workflow test",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Classify
        score = classifier.classify(data, "json", {"source": "test.com"})
        
        # Verify score
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.overall_score, 0.0)
        self.assertLessEqual(score.overall_score, 1.0)
        self.assertIn(score.action, ["accept", "warn", "quarantine", "reject"])


def run_tests():
    """Run all test suites"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDataQualityClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestRescoreSchedules))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)