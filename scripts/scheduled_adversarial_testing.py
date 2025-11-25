#!/usr/bin/env python3
"""
Scheduled Adversarial Testing - Phase 3 Implementation (FIXED)

This script provides a cron-compatible wrapper for running AdversarialValidator attacks
on a scheduled basis. It can be called directly or via cron to perform automated
security testing of the VULCAN system.

Usage:
    # Run with default config
    python scripts/scheduled_adversarial_testing.py

    # Run with custom config
    python scripts/scheduled_adversarial_testing.py --config configs/adversarial_testing_schedule.json

    # Run specific attack types
    python scripts/scheduled_adversarial_testing.py --attacks fgsm,pgd

    # Dry run (don't execute attacks, just show what would run)
    python scripts/scheduled_adversarial_testing.py --dry-run

Cron Example:
    # Run every day at 2 AM
    0 2 * * * cd /path/to/VulcanAMI_LLM && python scripts/scheduled_adversarial_testing.py

    # Run every 6 hours
    0 */6 * * * cd /path/to/VulcanAMI_LLM && python scripts/scheduled_adversarial_testing.py
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import platform

# Import signal only on Unix-like systems
if platform.system() != 'Windows':
    import signal
    SIGNAL_AVAILABLE = True
else:
    SIGNAL_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.vulcan.safety.adversarial_formal import (
        AdversarialValidator,
        initialize_adversarial
    )
    ADVERSARIAL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AdversarialValidator not available: {e}")
    ADVERSARIAL_AVAILABLE = False

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduled_adversarial_testing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScheduledAdversarialTester:
    """Wrapper for running AdversarialValidator on a schedule."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize scheduled tester.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.validator = None
        self.results = []
        
        if ADVERSARIAL_AVAILABLE:
            try:
                self.validator = initialize_adversarial()
                logger.info("AdversarialValidator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AdversarialValidator: {e}")
                self.validator = None
        else:
            logger.error("AdversarialValidator not available - cannot run tests")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'attack_types': ['fgsm', 'pgd', 'semantic'],
            'epsilon_values': [0.01, 0.05, 0.1],
            'num_samples': 100,
            'timeout_seconds': 3600,
            'save_results': True,
            'results_dir': 'logs/adversarial_testing',
            'alert_on_failure': True,
            'failure_threshold': 0.8
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _extract_report_data(self, report: Any) -> Dict[str, Any]:
        """Extract data from SafetyReport or similar object.
        
        Args:
            report: Report object from validator
            
        Returns:
            Dictionary with report data
        """
        # Try different methods to extract data from the report object
        report_dict = {}
        
        # If it's already a dictionary, return it
        if isinstance(report, dict):
            return report
        
        # Try to convert to dictionary using various methods
        if hasattr(report, 'to_dict'):
            try:
                return report.to_dict()
            except:
                pass
        
        if hasattr(report, '__dict__'):
            try:
                # Get all non-private attributes
                report_dict = {k: v for k, v in vars(report).items() 
                             if not k.startswith('_')}
                return report_dict
            except:
                pass
        
        # Extract known attributes manually
        known_attributes = [
            'safe', 'robustness_score', 'violations', 'attack_results',
            'passed', 'failed', 'error', 'message', 'details',
            'adversarial_examples', 'success_rate', 'avg_perturbation'
        ]
        
        for attr in known_attributes:
            if hasattr(report, attr):
                try:
                    value = getattr(report, attr)
                    # Convert non-serializable objects to strings
                    if hasattr(value, '__dict__') and not isinstance(value, (dict, list, str, int, float, bool)):
                        value = str(value)
                    report_dict[attr] = value
                except:
                    pass
        
        # If we couldn't extract anything meaningful, just convert to string
        if not report_dict:
            report_dict = {'report_str': str(report), 'type': type(report).__name__}
        
        return report_dict
    
    def run_attack(self, attack_type: str, epsilon: float) -> Dict[str, Any]:
        """Run a single attack.
        
        Args:
            attack_type: Type of attack ('fgsm', 'pgd', 'semantic')
            epsilon: Perturbation strength
            
        Returns:
            Attack results dictionary
        """
        if not self.validator:
            return {
                'attack_type': attack_type,
                'epsilon': epsilon,
                'success': False,
                'error': 'Validator not available'
            }
        
        logger.info(f"Running {attack_type} attack with epsilon={epsilon}")
        start_time = time.time()
        
        try:
            # Create a mock action and context for testing
            test_action = {
                'type': 'test_action',
                'parameters': {'test': True}
            }
            test_context = {
                'test_mode': True,
                'scheduled': True,
                'attack_type': attack_type,
                'epsilon': epsilon
            }
            
            # Mock validator function - must return (safe, reason, confidence) tuple
            def mock_validator(action, context):
                return (True, "OK", 0.9)
            
            # Try to run the attack with error handling
            report = None
            error_msg = None
            
            try:
                # Attempt to run validate_robustness
                report = self.validator.validate_robustness(
                    test_action,
                    test_context,
                    mock_validator
                )
            except TypeError as e:
                # Handle parameter mismatch
                if "unpack" in str(e):
                    # Skip trojan attacks that have the unpacking issue
                    if attack_type == 'trojan':
                        logger.warning(f"Skipping trojan attack due to known issue: {e}")
                        error_msg = f"Trojan attack not supported: {e}"
                    else:
                        # Try with different parameters
                        try:
                            report = self.validator.validate_robustness(
                                mock_validator,
                                test_action,
                                test_context
                            )
                        except:
                            # Try without the validator function
                            try:
                                report = self.validator.validate_robustness(
                                    test_action,
                                    test_context
                                )
                            except Exception as e2:
                                error_msg = f"Failed with multiple parameter combinations: {e2}"
                else:
                    error_msg = str(e)
            except AttributeError as e:
                # Handle missing methods
                if "validate_robustness" in str(e):
                    # Try alternative method names
                    if hasattr(self.validator, 'run_attack'):
                        report = self.validator.run_attack(attack_type, epsilon)
                    elif hasattr(self.validator, 'validate'):
                        report = self.validator.validate(test_action, test_context)
                    else:
                        error_msg = f"No suitable validation method found: {e}"
                else:
                    error_msg = str(e)
            
            if error_msg:
                logger.error(f"Attack execution error: {error_msg}")
                return {
                    'attack_type': attack_type,
                    'epsilon': epsilon,
                    'success': False,
                    'error': error_msg,
                    'duration_seconds': time.time() - start_time,
                    'timestamp': datetime.now().isoformat()
                }
            
            duration = time.time() - start_time
            
            # Convert report to dictionary
            report_dict = self._extract_report_data(report) if report else {}
            
            result = {
                'attack_type': attack_type,
                'epsilon': epsilon,
                'success': True,
                'duration_seconds': duration,
                'report': report_dict,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if we should alert on this result
            if self.config.get('alert_on_failure'):
                self._check_and_alert(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Attack failed: {e}")
            return {
                'attack_type': attack_type,
                'epsilon': epsilon,
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def run_scheduled_tests(self, attack_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all scheduled adversarial tests.
        
        Args:
            attack_types: List of attack types to run (None = use config)
            
        Returns:
            Summary of test results
        """
        if not self.validator:
            logger.error("Cannot run tests - validator not available")
            return {
                'success': False,
                'error': 'AdversarialValidator not available',
                'results': []
            }
        
        logger.info("Starting scheduled adversarial testing")
        start_time = time.time()
        
        # Get attack types from config or argument
        attacks = attack_types or self.config['attack_types']
        epsilons = self.config['epsilon_values']
        
        # Set up timeout
        timeout = self.config.get('timeout_seconds', 3600)
        timeout_end = start_time + timeout
        
        # Setup timeout handling
        if SIGNAL_AVAILABLE:
            def timeout_handler(signum, frame):
                raise TimeoutError("Test suite exceeded time limit")
            
            # Set timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        else:
            # On Windows, we'll use a manual timeout check
            logger.warning("Signal-based timeout not available on Windows, using manual timeout check")
        
        try:
            # Run all combinations of attacks and epsilon values
            self.results = []
            for attack_type in attacks:
                for epsilon in epsilons:
                    # Check timeout manually
                    if time.time() > timeout_end:
                        raise TimeoutError("Test suite exceeded time limit")
                    
                    result = self.run_attack(attack_type, epsilon)
                    self.results.append(result)
            
            duration = time.time() - start_time
            
            # Generate summary
            summary = {
                'success': True,
                'total_tests': len(self.results),
                'successful_tests': sum(1 for r in self.results if r.get('success')),
                'failed_tests': sum(1 for r in self.results if not r.get('success')),
                'total_duration_seconds': duration,
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }
            
            logger.info(f"Completed {summary['total_tests']} tests in {duration:.2f}s")
            logger.info(f"Success: {summary['successful_tests']}, Failed: {summary['failed_tests']}")
            
            # Save results if configured
            if self.config.get('save_results'):
                self._save_results(summary)
            
            return summary
            
        except TimeoutError as e:
            logger.error(f"Test suite timed out: {e}")
            # Save partial results
            if self.results and self.config.get('save_results'):
                partial_summary = {
                    'success': False,
                    'error': 'Timeout',
                    'total_tests': len(self.results),
                    'results': self.results,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_results(partial_summary)
            raise
        finally:
            # Cancel the alarm (only on Unix-like systems)
            if SIGNAL_AVAILABLE:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    
    def _check_and_alert(self, result: Dict[str, Any]):
        """Check result and send alert if needed.
        
        Args:
            result: Test result dictionary
        """
        # Extract relevant metrics from report
        report = result.get('report', {})
        
        # Try to get robustness score from various possible locations
        robustness = None
        if isinstance(report, dict):
            # Try different possible keys
            for key in ['robustness_score', 'score', 'success_rate', 'safety_score']:
                if key in report:
                    robustness = report[key]
                    break
        
        # If we couldn't find a score, try to infer from other fields
        if robustness is None and isinstance(report, dict):
            if 'safe' in report:
                robustness = 1.0 if report['safe'] else 0.0
            elif 'passed' in report and 'failed' in report:
                total = report.get('passed', 0) + report.get('failed', 0)
                if total > 0:
                    robustness = report.get('passed', 0) / total
        
        threshold = self.config.get('failure_threshold', 0.8)
        
        if robustness is not None and robustness < threshold:
            logger.warning(f"ALERT: Robustness below threshold! "
                          f"Attack: {result['attack_type']}, "
                          f"Epsilon: {result['epsilon']}, "
                          f"Score: {robustness:.3f}")
            
            # In a production system, this would send alerts via Slack, email, etc.
            # For now, we just log the alert
            self._send_alert(result, robustness)
    
    def _send_alert(self, result: Dict[str, Any], robustness: float):
        """Send alert for failed test (placeholder for production alerting).
        
        Args:
            result: Test result that triggered alert
            robustness: Robustness score that failed threshold
        """
        alert_message = {
            'severity': 'WARNING',
            'component': 'AdversarialTesting',
            'timestamp': datetime.now().isoformat(),
            'attack_type': result['attack_type'],
            'epsilon': result['epsilon'],
            'robustness_score': robustness,
            'threshold': self.config.get('failure_threshold', 0.8),
            'message': f"Adversarial attack succeeded with score {robustness:.3f}"
        }
        
        # Log alert to separate file
        alert_log_path = Path('logs') / 'adversarial_alerts.jsonl'
        try:
            with open(alert_log_path, 'a') as f:
                f.write(json.dumps(alert_message) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save test results to file.
        
        Args:
            summary: Test summary dictionary
        """
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'adversarial_test_results_{timestamp}.json'
        
        try:
            # Make sure results are JSON serializable
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(v) for v in obj]
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return obj
            
            serializable_summary = make_serializable(summary)
            
            with open(filename, 'w') as f:
                json.dump(serializable_summary, f, indent=2, default=str)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            # Try to save a minimal version
            try:
                minimal_summary = {
                    'timestamp': summary.get('timestamp'),
                    'total_tests': summary.get('total_tests'),
                    'successful_tests': summary.get('successful_tests'),
                    'failed_tests': summary.get('failed_tests'),
                    'error': f'Full results could not be saved: {e}'
                }
                with open(filename, 'w') as f:
                    json.dump(minimal_summary, f, indent=2)
                logger.info(f"Minimal results saved to {filename}")
            except Exception as e2:
                logger.error(f"Failed to save even minimal results: {e2}")


def main():
    """Main entry point for scheduled testing."""
    parser = argparse.ArgumentParser(
        description='Run scheduled adversarial testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--attacks',
        type=str,
        help='Comma-separated list of attack types (fgsm,pgd,semantic)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would run without executing'
    )
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = ScheduledAdversarialTester(config_path=args.config)
    
    # Parse attack types if provided
    attack_types = None
    if args.attacks:
        attack_types = [a.strip() for a in args.attacks.split(',')]
    
    if args.dry_run:
        logger.info("DRY RUN - No attacks will be executed")
        logger.info(f"Configuration: {json.dumps(tester.config, indent=2)}")
        attacks = attack_types or tester.config['attack_types']
        logger.info(f"Would run attacks: {attacks}")
        logger.info(f"With epsilon values: {tester.config['epsilon_values']}")
        return 0
    
    # Run tests
    try:
        summary = tester.run_scheduled_tests(attack_types=attack_types)
        
        # Exit with error code if tests failed
        if summary['failed_tests'] > 0:
            logger.error(f"{summary['failed_tests']} tests failed")
            return 1
        
        logger.info("All tests completed successfully")
        return 0
        
    except TimeoutError as e:
        logger.error(f"Test suite timed out: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
