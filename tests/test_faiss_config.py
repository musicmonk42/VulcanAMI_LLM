"""
Tests for FAISS Configuration Module

Tests proper initialization, CPU capability detection, thread safety,
and fallback behavior of the FAISS configuration system.
"""

import sys
import threading
import unittest
from unittest.mock import MagicMock, patch


class TestFAISSConfig(unittest.TestCase):
    """Test suite for FAISS configuration module"""
    
    def setUp(self):
        """Reset FAISS config state before each test"""
        # Import and reset module state
        import src.utils.faiss_config as faiss_config
        faiss_config.FAISS_AVAILABLE = False
        faiss_config.FAISS_MODULE = None
        faiss_config.FAISS_INSTRUCTION_SET = None
    
    @patch('src.utils.faiss_config.warnings')
    def test_initialize_faiss_success(self, mock_warnings):
        """Test successful FAISS initialization"""
        # Mock faiss module
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            from src.utils.faiss_config import initialize_faiss
            
            faiss_mod, available, instr_set = initialize_faiss()
            
            # Should succeed
            self.assertIsNotNone(faiss_mod)
            self.assertTrue(available)
            # Warning filters should be set
            self.assertTrue(mock_warnings.filterwarnings.called)
    
    def test_initialize_faiss_not_available(self):
        """Test FAISS initialization when module not available"""
        # Remove faiss from sys.modules if present
        faiss_backup = sys.modules.pop('faiss', None)
        
        try:
            # Mock import to raise ImportError
            with patch('builtins.__import__', side_effect=ImportError("No module named 'faiss'")):
                from src.utils.faiss_config import initialize_faiss
                
                faiss_mod, available, instr_set = initialize_faiss()
                
                # Should fail gracefully
                self.assertIsNone(faiss_mod)
                self.assertFalse(available)
                self.assertIsNone(instr_set)
        finally:
            # Restore faiss if it was present
            if faiss_backup:
                sys.modules['faiss'] = faiss_backup
    
    @patch('src.utils.faiss_config.warnings')
    def test_get_faiss(self, mock_warnings):
        """Test get_faiss convenience function"""
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            from src.utils.faiss_config import get_faiss
            
            faiss_mod = get_faiss()
            
            # Should return the module
            self.assertIsNotNone(faiss_mod)
    
    @patch('src.utils.faiss_config.warnings')
    def test_is_faiss_available(self, mock_warnings):
        """Test is_faiss_available function"""
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            from src.utils.faiss_config import is_faiss_available
            
            available = is_faiss_available()
            
            # Should be available
            self.assertTrue(available)
    
    @patch('src.utils.faiss_config.warnings')
    def test_get_faiss_instruction_set(self, mock_warnings):
        """Test instruction set detection"""
        mock_faiss = MagicMock()
        
        # Mock CPU capabilities
        mock_caps = MagicMock()
        mock_caps.get_best_vector_instruction_set.return_value = "AVX2"
        mock_caps.get_performance_tier.return_value = "Medium Performance"
        mock_caps.has_avx512f = False
        mock_caps.has_avx2 = True
        mock_caps.has_avx = True
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            with patch('src.utils.faiss_config.get_cpu_capabilities', return_value=mock_caps):
                from src.utils.faiss_config import get_faiss_instruction_set, initialize_faiss
                
                # Force re-initialization
                import src.utils.faiss_config as fc
                fc.FAISS_MODULE = None
                
                initialize_faiss()
                instr_set = get_faiss_instruction_set()
                
                # Should detect AVX2
                self.assertEqual(instr_set, "AVX2")
    
    @patch('src.utils.faiss_config.warnings')
    def test_thread_safety(self, mock_warnings):
        """Test thread-safe initialization"""
        mock_faiss = MagicMock()
        init_count = {'value': 0}
        
        def mock_import(name, *args, **kwargs):
            if name == 'faiss':
                init_count['value'] += 1
                return mock_faiss
            return __import__(name, *args, **kwargs)
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            # Reset module state
            import src.utils.faiss_config as fc
            fc.FAISS_MODULE = None
            fc.FAISS_AVAILABLE = False
            
            from src.utils.faiss_config import initialize_faiss
            
            # Run multiple threads
            threads = []
            results = []
            
            def init_in_thread():
                result = initialize_faiss()
                results.append(result)
            
            for _ in range(10):
                t = threading.Thread(target=init_in_thread)
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            # All threads should get same result
            self.assertEqual(len(results), 10)
            first_result = results[0]
            for result in results[1:]:
                self.assertEqual(result[1], first_result[1])  # availability should match
    
    @patch('src.utils.faiss_config.warnings')
    def test_get_faiss_config_info(self, mock_warnings):
        """Test comprehensive config info retrieval"""
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            from src.utils.faiss_config import get_faiss_config_info
            
            config = get_faiss_config_info()
            
            # Should have required keys
            self.assertIn('available', config)
            self.assertIn('instruction_set', config)
            self.assertIn('cpu_capabilities', config)
            self.assertIn('recommendations', config)
            self.assertIsInstance(config['recommendations'], list)
    
    def test_cpu_capability_detection_fallback(self):
        """Test fallback when CPU capability detection fails"""
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            # Mock cpu_capabilities to not be available
            with patch('src.utils.faiss_config.get_cpu_capabilities', side_effect=ImportError):
                from src.utils.faiss_config import initialize_faiss
                
                # Reset state
                import src.utils.faiss_config as fc
                fc.FAISS_MODULE = None
                
                faiss_mod, available, instr_set = initialize_faiss()
                
                # Should still succeed, but instruction set unknown
                self.assertTrue(available)
                self.assertEqual(instr_set, "UNKNOWN")
    
    @patch('src.utils.faiss_config.warnings')
    def test_warning_suppression(self, mock_warnings):
        """Test that swigfaiss_avx512 warnings are suppressed"""
        mock_faiss = MagicMock()
        
        with patch.dict('sys.modules', {'faiss': mock_faiss}):
            from src.utils.faiss_config import initialize_faiss
            
            # Reset state
            import src.utils.faiss_config as fc
            fc.FAISS_MODULE = None
            
            initialize_faiss()
            
            # Verify filterwarnings was called with swigfaiss_avx512 pattern
            self.assertTrue(mock_warnings.filterwarnings.called)
            
            # Check at least one call has the expected pattern
            found_pattern = False
            for call in mock_warnings.filterwarnings.call_args_list:
                args, kwargs = call
                if 'message' in kwargs and 'swigfaiss_avx512' in str(kwargs['message']):
                    found_pattern = True
                    break
            
            self.assertTrue(found_pattern, "Should suppress swigfaiss_avx512 warnings")


class TestFAISSConfigIntegration(unittest.TestCase):
    """Integration tests for FAISS configuration"""
    
    def test_real_initialization(self):
        """Test with real FAISS if available"""
        try:
            import faiss
            faiss_available = True
        except ImportError:
            faiss_available = False
        
        from src.utils.faiss_config import initialize_faiss, is_faiss_available
        
        # Reset state
        import src.utils.faiss_config as fc
        fc.FAISS_MODULE = None
        fc.FAISS_AVAILABLE = False
        
        faiss_mod, available, instr_set = initialize_faiss()
        
        # Should match actual availability
        self.assertEqual(available, faiss_available)
        self.assertEqual(is_faiss_available(), faiss_available)
        
        if available:
            self.assertIsNotNone(faiss_mod)
            self.assertIsNotNone(instr_set)


if __name__ == '__main__':
    unittest.main()
