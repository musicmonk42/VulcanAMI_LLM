"""
Tests for Scheduled Adversarial Testing - Phase 3

Tests the scheduled testing wrapper in scripts/scheduled_adversarial_testing.py:
- Configuration loading
- Test scheduling and execution
- Result saving and reporting
- Alert generation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add scripts directory to path for testing
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

from scheduled_adversarial_testing import ScheduledAdversarialTester


class TestConfigurationLoading:
    """Test configuration loading and defaults."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        tester = ScheduledAdversarialTester(config_path=None)

        assert 'attack_types' in tester.config
        assert 'epsilon_values' in tester.config
        assert 'save_results' in tester.config
        assert isinstance(tester.config['attack_types'], list)

    def test_load_custom_config(self):
        """Test loading custom configuration from file."""
        custom_config = {
            'attack_types': ['fgsm'],
            'epsilon_values': [0.1],
            'num_samples': 50
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)

            assert tester.config['attack_types'] == ['fgsm']
            assert tester.config['epsilon_values'] == [0.1]
            assert tester.config['num_samples'] == 50
        finally:
            os.unlink(config_path)

    def test_load_nonexistent_config(self):
        """Test that nonexistent config falls back to defaults."""
        tester = ScheduledAdversarialTester(config_path='/nonexistent/config.json')

        # Should use defaults
        assert 'attack_types' in tester.config
        assert len(tester.config['attack_types']) > 0

    def test_config_defaults(self):
        """Test that all required config keys have defaults."""
        tester = ScheduledAdversarialTester()

        required_keys = [
            'attack_types',
            'epsilon_values',
            'num_samples',
            'timeout_seconds',
            'save_results',
            'results_dir',
            'alert_on_failure',
            'failure_threshold'
        ]

        for key in required_keys:
            assert key in tester.config, f"Missing required config key: {key}"


class TestAttackExecution:
    """Test attack execution functionality."""

    def test_run_attack_success(self):
        """Test successful attack execution."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.85,
            'attack_success': False
        })

        tester = ScheduledAdversarialTester()
        tester.validator = mock_validator  # Ensure validator is set
        result = tester.run_attack('fgsm', 0.1)

        assert result['success'] is True
        assert result['attack_type'] == 'fgsm'
        assert result['epsilon'] == 0.1
        assert 'duration_seconds' in result
        assert 'timestamp' in result

    def test_run_attack_no_validator(self):
        """Test attack execution when validator is unavailable."""
        tester = ScheduledAdversarialTester()
        tester.validator = None

        result = tester.run_attack('fgsm', 0.1)

        assert result['success'] is False
        assert 'error' in result
        assert result['attack_type'] == 'fgsm'

    def test_run_attack_exception(self):
        """Test attack execution with exception."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(side_effect=Exception("Test error"))

        tester = ScheduledAdversarialTester()
        tester.validator = mock_validator
        result = tester.run_attack('fgsm', 0.1)

        assert result['success'] is False
        assert 'error' in result
        assert 'Test error' in result['error']


class TestScheduledTestExecution:
    """Test full scheduled test execution."""

    def test_run_scheduled_tests_success(self):
        """Test running all scheduled tests successfully."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.85,
            'attack_success': False
        })

        config = {
            'attack_types': ['fgsm', 'pgd'],
            'epsilon_values': [0.1, 0.2],
            'save_results': False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)
            tester.validator = mock_validator
            summary = tester.run_scheduled_tests()

            assert summary['success'] is True
            assert summary['total_tests'] == 4  # 2 attacks * 2 epsilons
            assert summary['successful_tests'] == 4
            assert summary['failed_tests'] == 0
            assert 'total_duration_seconds' in summary
            assert 'timestamp' in summary
        finally:
            os.unlink(config_path)

    def test_run_scheduled_tests_no_validator(self):
        """Test scheduled tests when validator is unavailable."""
        tester = ScheduledAdversarialTester()
        tester.validator = None

        summary = tester.run_scheduled_tests()

        assert summary['success'] is False
        assert 'error' in summary

    def test_run_scheduled_tests_custom_attacks(self):
        """Test running tests with custom attack types."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.85
        })

        tester = ScheduledAdversarialTester()
        tester.validator = mock_validator
        tester.config['save_results'] = False

        summary = tester.run_scheduled_tests(attack_types=['fgsm'])

        assert summary['success'] is True
        # Should run fgsm with all epsilon values
        assert summary['total_tests'] == len(tester.config['epsilon_values'])


class TestResultSaving:
    """Test result saving functionality."""

    def test_save_results(self):
        """Test saving results to file."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.85
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'attack_types': ['fgsm'],
                'epsilon_values': [0.1],
                'save_results': True,
                'results_dir': tmpdir
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_path = f.name

            try:
                tester = ScheduledAdversarialTester(config_path=config_path)
                tester.validator = mock_validator
                summary = tester.run_scheduled_tests()

                # Check that results file was created
                results_files = list(Path(tmpdir).glob('adversarial_test_results_*.json'))
                assert len(results_files) == 1

                # Verify contents
                with open(results_files[0]) as f:
                    saved_summary = json.load(f)

                assert saved_summary['success'] is True
                assert saved_summary['total_tests'] == 1
            finally:
                os.unlink(config_path)

    def test_no_save_results(self):
        """Test that results are not saved when disabled."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.85
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'attack_types': ['fgsm'],
                'epsilon_values': [0.1],
                'save_results': False,
                'results_dir': tmpdir
            }

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config, f)
                config_path = f.name

            try:
                tester = ScheduledAdversarialTester(config_path=config_path)
                tester.validator = mock_validator
                summary = tester.run_scheduled_tests()

                # Check that no results file was created
                results_files = list(Path(tmpdir).glob('adversarial_test_results_*.json'))
                assert len(results_files) == 0
            finally:
                os.unlink(config_path)


class TestAlertGeneration:
    """Test alert generation functionality."""

    def test_alert_on_low_robustness(self):
        """Test that alert is generated for low robustness."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.5  # Below default threshold of 0.8
        })

        config = {
            'attack_types': ['fgsm'],
            'epsilon_values': [0.1],
            'alert_on_failure': True,
            'failure_threshold': 0.8,
            'save_results': False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)
            tester.validator = mock_validator

            # Capture log output to verify alert was logged
            with patch('scheduled_adversarial_testing.logger') as mock_logger:
                summary = tester.run_scheduled_tests()

                # Verify that warning was logged
                assert mock_logger.warning.called
                warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                assert any('ALERT' in str(call) for call in warning_calls)
        finally:
            os.unlink(config_path)

    def test_no_alert_on_high_robustness(self):
        """Test that no alert is generated for high robustness."""
        mock_validator = Mock()
        mock_validator.validate_robustness = Mock(return_value={
            'robustness_score': 0.95  # Above threshold
        })

        config = {
            'attack_types': ['fgsm'],
            'epsilon_values': [0.1],
            'alert_on_failure': True,
            'failure_threshold': 0.8,
            'save_results': False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)
            tester.validator = mock_validator

            with patch('scheduled_adversarial_testing.logger') as mock_logger:
                summary = tester.run_scheduled_tests()

                # Verify that no warning was logged for low robustness
                if mock_logger.warning.called:
                    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
                    assert not any('ALERT' in str(call) for call in warning_calls)
        finally:
            os.unlink(config_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_attack_types(self):
        """Test handling of empty attack types."""
        config = {
            'attack_types': [],
            'epsilon_values': [0.1],
            'save_results': False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)
            tester.validator = None  # Simulate no validator

            summary = tester.run_scheduled_tests()

            # Should handle gracefully
            assert 'error' in summary or summary['total_tests'] == 0
        finally:
            os.unlink(config_path)

    def test_empty_epsilon_values(self):
        """Test handling of empty epsilon values."""
        config = {
            'attack_types': ['fgsm'],
            'epsilon_values': [],
            'save_results': False
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            tester = ScheduledAdversarialTester(config_path=config_path)
            tester.validator = None  # Simulate no validator

            summary = tester.run_scheduled_tests()

            # Should handle gracefully
            assert 'error' in summary or summary['total_tests'] == 0
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
