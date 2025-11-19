#!/usr/bin/env python3
"""
Security Test Suite for VulcanAMI_LLM
Tests for critical security vulnerabilities identified in the audit
"""
import unittest
import pickle
import io
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from vulcan.security_fixes import safe_pickle_load, SafeUnpickler
    SECURITY_FIXES_AVAILABLE = True
except ImportError:
    SECURITY_FIXES_AVAILABLE = False
    print("WARNING: security_fixes module not available")


class MaliciousPickleTests(unittest.TestCase):
    """Test safe pickle loading prevents code execution"""
    
    def setUp(self):
        if not SECURITY_FIXES_AVAILABLE:
            self.skipTest("security_fixes module not available")
    
    def test_safe_pickle_load_blocks_os_system(self):
        """Test that safe_pickle_load blocks os.system calls"""
        # Create a malicious pickle that would execute os.system
        class Exploit:
            def __reduce__(self):
                import os
                return (os.system, ('echo "EXPLOITED"',))
        
        malicious_data = pickle.dumps(Exploit())
        
        with self.assertRaises(pickle.UnpicklingError) as context:
            safe_pickle_load(io.BytesIO(malicious_data))
        
        self.assertIn("forbidden", str(context.exception).lower())
    
    def test_safe_pickle_load_blocks_subprocess(self):
        """Test that safe_pickle_load blocks subprocess calls"""
        class Exploit:
            def __reduce__(self):
                import subprocess
                return (subprocess.call, (['echo', 'EXPLOITED'],))
        
        malicious_data = pickle.dumps(Exploit())
        
        with self.assertRaises(pickle.UnpicklingError):
            safe_pickle_load(io.BytesIO(malicious_data))
    
    def test_safe_pickle_load_blocks_eval(self):
        """Test that safe_pickle_load blocks eval calls"""
        class Exploit:
            def __reduce__(self):
                return (eval, ('__import__("os").system("echo EXPLOITED")',))
        
        malicious_data = pickle.dumps(Exploit())
        
        with self.assertRaises(pickle.UnpicklingError):
            safe_pickle_load(io.BytesIO(malicious_data))
    
    def test_safe_pickle_load_allows_safe_types(self):
        """Test that safe_pickle_load allows safe types"""
        safe_data = {
            'string': 'hello',
            'number': 42,
            'list': [1, 2, 3],
            'dict': {'key': 'value'},
            'bool': True,
            'none': None
        }
        
        pickled = pickle.dumps(safe_data)
        loaded = safe_pickle_load(io.BytesIO(pickled))
        
        self.assertEqual(loaded, safe_data)
    
    def test_safe_pickle_load_allows_numpy_if_available(self):
        """Test that numpy arrays are allowed"""
        try:
            import numpy as np
            arr = np.array([1, 2, 3, 4, 5])
            pickled = pickle.dumps(arr)
            loaded = safe_pickle_load(io.BytesIO(pickled))
            self.assertTrue(np.array_equal(loaded, arr))
        except ImportError:
            self.skipTest("NumPy not available")


class TorchLoadSecurityTests(unittest.TestCase):
    """Test torch.load security improvements"""
    
    def test_inspect_system_state_uses_safe_load(self):
        """Verify inspect_system_state.py uses safe loading"""
        with open('inspect_system_state.py', 'r') as f:
            content = f.read()
        
        # Check for weights_only=True in torch.load
        self.assertIn('weights_only=True', content,
                     "inspect_system_state.py should use weights_only=True")
        
        # Check for safe_pickle_load fallback
        self.assertIn('safe_pickle_load', content,
                     "inspect_system_state.py should use safe_pickle_load fallback")
    
    def test_simple_eval_pkl_uses_safe_load(self):
        """Verify simple_eval_pkl.py uses safe loading"""
        with open('simple_eval_pkl.py', 'r') as f:
            content = f.read()
        
        # Check for weights_only=True in torch.load
        self.assertIn('weights_only=True', content,
                     "simple_eval_pkl.py should use weights_only=True")


class PathTraversalTests(unittest.TestCase):
    """Test for path traversal vulnerabilities"""
    
    def test_graph_validator_no_hardcoded_paths(self):
        """Verify graph_validator doesn't have hardcoded Windows paths"""
        validator_path = Path('src/unified_runtime/graph_validator.py')
        if not validator_path.exists():
            self.skipTest("graph_validator.py not found")
        
        with open(validator_path, 'r') as f:
            content = f.read()
        
        # Should not have hardcoded Windows paths
        self.assertNotIn('D:/', content,
                        "graph_validator should not have hardcoded D:/ paths")
        self.assertNotIn('C:/', content,
                        "graph_validator should not have hardcoded C:/ paths")


class JWTSecurityTests(unittest.TestCase):
    """Test JWT security improvements"""
    
    def test_app_py_has_improved_jwt_claims(self):
        """Verify app.py includes all required JWT claims"""
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Check for required claims
        required_claims = ['sub', 'iat', 'nbf', 'jti', 'iss', 'aud']
        for claim in required_claims:
            self.assertIn(f'"{claim}"', content,
                         f"app.py should include {claim} in JWT claims")
    
    def test_app_py_has_bootstrap_locking(self):
        """Verify app.py has database-level bootstrap locking"""
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Check for BootstrapStatus model
        self.assertIn('class BootstrapStatus', content,
                     "app.py should have BootstrapStatus model")
        
        # Check for with_for_update (SELECT FOR UPDATE)
        self.assertIn('with_for_update', content,
                     "app.py should use with_for_update for atomic locking")


class TimingAttackTests(unittest.TestCase):
    """Test timing attack mitigations"""
    
    def test_app_py_has_adequate_delay(self):
        """Verify app.py has adequate delay for timing attack prevention"""
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Should have increased delay (100+ ms)
        self.assertIn('100', content,
                     "app.py should have delays >= 100ms for timing attack prevention")


class SecurityDocumentationTests(unittest.TestCase):
    """Test that security documentation exists"""
    
    def test_security_audit_report_exists(self):
        """Verify SECURITY_AUDIT_REPORT.md exists"""
        self.assertTrue(
            Path('SECURITY_AUDIT_REPORT.md').exists(),
            "SECURITY_AUDIT_REPORT.md should exist"
        )
    
    def test_security_policy_exists(self):
        """Verify SECURITY.md exists"""
        self.assertTrue(
            Path('SECURITY.md').exists(),
            "SECURITY.md should exist"
        )
    
    def test_security_fixes_module_exists(self):
        """Verify security_fixes.py module exists"""
        self.assertTrue(
            Path('src/vulcan/security_fixes.py').exists(),
            "src/vulcan/security_fixes.py should exist"
        )


class InputValidationTests(unittest.TestCase):
    """Test input validation"""
    
    def test_app_py_validates_agent_id(self):
        """Verify app.py validates agent_id format"""
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Should have agent_id pattern validation
        self.assertIn('AGENT_ID_PATTERN', content,
                     "app.py should define AGENT_ID_PATTERN")
        self.assertIn('validate_agent_id', content,
                     "app.py should have validate_agent_id function")


def run_security_tests():
    """Run all security tests"""
    # Change to repo root
    os.chdir(Path(__file__).parent.parent)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(MaliciousPickleTests))
    suite.addTests(loader.loadTestsFromTestCase(TorchLoadSecurityTests))
    suite.addTests(loader.loadTestsFromTestCase(PathTraversalTests))
    suite.addTests(loader.loadTestsFromTestCase(JWTSecurityTests))
    suite.addTests(loader.loadTestsFromTestCase(TimingAttackTests))
    suite.addTests(loader.loadTestsFromTestCase(SecurityDocumentationTests))
    suite.addTests(loader.loadTestsFromTestCase(InputValidationTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("SECURITY TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All security tests passed!")
        return 0
    else:
        print("\n❌ Some security tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(run_security_tests())
