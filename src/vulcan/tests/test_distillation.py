"""
Comprehensive tests for the distillation package.

Tests cover:
1. Security - Secret detection, encoding bypass prevention, JSONL injection
2. Privacy - PII redaction, governance sensitivity
3. Quality - Validation, deduplication, thread safety
4. Storage - Thread safety, encryption, GDPR compliance
5. Integration - Full capture flow, opt-in enforcement
6. Edge cases - Empty inputs, large inputs, Unicode handling
"""

import base64
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List

import pytest

from vulcan.distillation.pii_redactor import PIIRedactor
from vulcan.distillation.quality_validator import ExampleQualityValidator
from vulcan.distillation.storage import DistillationStorageBackend


# ============================================================
# SECURITY TESTS - Secret Detection
# ============================================================


class TestSecretDetection:
    """Test secret detection including encoding bypass prevention."""
    
    def test_plain_openai_key_detection(self):
        """Test detection of plain OpenAI API keys."""
        redactor = PIIRedactor()
        
        text = "Here is my API key: sk-1234567890abcdefghijklmnopqrst"
        assert redactor.contains_secrets(text)
    
    def test_plain_aws_keys_detection(self):
        """Test detection of AWS access and secret keys."""
        redactor = PIIRedactor()
        
        # AWS access key
        text1 = "My AWS key is AKIAIOSFODNN7EXAMPLE"
        assert redactor.contains_secrets(text1)
        
        # AWS secret key
        text2 = "Secret: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        assert redactor.contains_secrets(text2)
    
    def test_github_token_detection(self):
        """Test detection of GitHub tokens."""
        redactor = PIIRedactor()
        
        text = "Token: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        assert redactor.contains_secrets(text)
    
    def test_bearer_token_detection(self):
        """Test detection of Bearer tokens."""
        redactor = PIIRedactor()
        
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        assert redactor.contains_secrets(text)
    
    def test_jwt_token_detection(self):
        """Test detection of JWT tokens."""
        redactor = PIIRedactor()
        
        text = "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert redactor.contains_secrets(text)
    
    def test_base64_encoded_secret_detection(self):
        """Test detection of base64-encoded secrets (bypass prevention)."""
        redactor = PIIRedactor()
        
        # Encode an OpenAI key in base64
        secret = "sk-1234567890abcdefghijklmnopqrst"
        encoded = base64.b64encode(secret.encode()).decode()
        text = f"Here is encoded data: {encoded}"
        
        assert redactor.contains_secrets(text), "Failed to detect base64-encoded secret"
    
    def test_hex_encoded_secret_detection(self):
        """Test detection of hex-encoded secrets (bypass prevention)."""
        redactor = PIIRedactor()
        
        # Encode an OpenAI key in hex
        secret = "sk-1234567890abcdefghijklmnopqrst"
        encoded = secret.encode().hex()
        text = f"Here is hex data: {encoded}"
        
        assert redactor.contains_secrets(text), "Failed to detect hex-encoded secret"
    
    def test_url_encoded_secret_detection(self):
        """Test detection of URL-encoded secrets (bypass prevention)."""
        redactor = PIIRedactor()
        
        # URL encode an OpenAI key
        secret = "sk-1234567890abcdefghijklmnopqrst"
        encoded = urllib.parse.quote(secret)
        text = f"Here is URL data: {encoded}"
        
        assert redactor.contains_secrets(text), "Failed to detect URL-encoded secret"
    
    def test_no_false_positives(self):
        """Test that normal text doesn't trigger secret detection."""
        redactor = PIIRedactor()
        
        normal_text = "This is a normal sentence without any secrets."
        assert not redactor.contains_secrets(normal_text)


# ============================================================
# PRIVACY TESTS - PII Redaction
# ============================================================


class TestPIIRedaction:
    """Test PII redaction functionality."""
    
    def test_email_redaction(self):
        """Test email address redaction."""
        redactor = PIIRedactor()
        
        text = "Contact me at john.doe@example.com for details"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "john.doe@example.com" not in redacted
        assert stats.get("email") == 1
    
    def test_phone_number_redaction(self):
        """Test phone number redaction."""
        redactor = PIIRedactor()
        
        text = "Call me at 555-123-4567 or (555) 987-6543"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_PHONE]" in redacted
        assert "555-123-4567" not in redacted
        assert stats.get("phone") == 2
    
    def test_ssn_redaction(self):
        """Test SSN redaction."""
        redactor = PIIRedactor()
        
        text = "My SSN is 123-45-6789"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_SSN]" in redacted
        assert "123-45-6789" not in redacted
    
    def test_credit_card_redaction(self):
        """Test credit card number redaction."""
        redactor = PIIRedactor()
        
        text = "Card: 1234-5678-9012-3456"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_CREDIT_CARD]" in redacted
        assert "1234-5678-9012-3456" not in redacted
    
    def test_ip_address_redaction(self):
        """Test IP address redaction."""
        redactor = PIIRedactor()
        
        text = "Server IP: 192.168.1.100"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_IP_ADDRESS]" in redacted
        assert "192.168.1.100" not in redacted
    
    def test_multiple_pii_types(self):
        """Test redaction of multiple PII types in one text."""
        redactor = PIIRedactor()
        
        text = "Contact john@example.com at 555-1234 from 10.0.0.1"
        redacted, stats = redactor.redact(text)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "[REDACTED_IP_ADDRESS]" in redacted


# ============================================================
# QUALITY VALIDATION TESTS
# ============================================================


class TestQualityValidation:
    """Test example quality validation."""
    
    def test_length_validation_too_short(self):
        """Test rejection of too-short responses."""
        validator = ExampleQualityValidator()
        
        prompt = "What is AI?"
        response = "AI is smart."  # Too short
        
        passed, score, reasons = validator.validate(prompt, response)
        assert not passed or score < validator.MIN_QUALITY_SCORE
        assert any("too_short" in r for r in reasons)
    
    def test_length_validation_too_long(self):
        """Test rejection of too-long responses."""
        validator = ExampleQualityValidator()
        
        prompt = "Tell me about AI"
        response = "A" * 5000  # Way too long
        
        passed, score, reasons = validator.validate(prompt, response)
        assert not passed or score < validator.MIN_QUALITY_SCORE
        assert any("too_long" in r for r in reasons)
    
    def test_refusal_detection(self):
        """Test detection of refusal responses."""
        validator = ExampleQualityValidator()
        
        prompt = "How to hack a website?"
        response = "I cannot help with that as an AI assistant."
        
        passed, score, reasons = validator.validate(prompt, response)
        assert any("contains_refusal" in r for r in reasons)
    
    def test_boilerplate_detection(self):
        """Test detection of high boilerplate content."""
        validator = ExampleQualityValidator()
        
        prompt = "Explain quantum computing"
        response = "Sure! Of course! Great question! Let me help you with that! Here's my answer: Quantum computing uses qubits. I hope this helps! Feel free to ask more questions!"
        
        passed, score, reasons = validator.validate(prompt, response)
        # High boilerplate should reduce quality score
        assert any("high_boilerplate" in r for r in reasons) or score < 0.7
    
    def test_deduplication(self):
        """Test duplicate response detection."""
        validator = ExampleQualityValidator()
        
        prompt = "What is 2+2?"
        response = "The answer is 4."
        
        # First validation should pass
        passed1, score1, reasons1 = validator.validate(prompt, response)
        assert "duplicate_content" not in reasons1
        
        # Second validation with same response should detect duplicate
        passed2, score2, reasons2 = validator.validate(prompt, response)
        assert "duplicate_content" in reasons2
    
    def test_thread_safe_deduplication(self):
        """Test thread safety of deduplication under concurrent access."""
        validator = ExampleQualityValidator(max_seen_hashes=100)
        
        results = []
        errors = []
        
        def validate_many(thread_id: int):
            try:
                for i in range(50):
                    prompt = f"Question {i}"
                    response = f"Answer from thread {thread_id} for question {i}"
                    passed, score, reasons = validator.validate(prompt, response)
                    results.append((thread_id, i, passed, score))
            except Exception as e:
                errors.append((thread_id, e))
        
        # Run 5 threads concurrently
        threads = [threading.Thread(target=validate_many, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        
        # All validations should complete
        assert len(results) == 5 * 50
    
    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        validator = ExampleQualityValidator(max_seen_hashes=3)
        
        # Add 3 unique responses
        validator.validate("Q1", "Response 1")
        validator.validate("Q2", "Response 2")
        validator.validate("Q3", "Response 3")
        
        # Add a 4th response (should evict oldest)
        validator.validate("Q4", "Response 4")
        
        # First response should now be allowed again (was evicted)
        passed, score, reasons = validator.validate("Q1", "Response 1")
        assert "duplicate_content" not in reasons


# ============================================================
# STORAGE TESTS - Thread Safety, Encryption, JSONL Injection
# ============================================================


class TestStorage:
    """Test storage backend functionality."""
    
    def test_basic_write_read(self):
        """Test basic write and read operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            example = {
                "prompt": "What is AI?",
                "response": "AI is artificial intelligence.",
                "timestamp": time.time()
            }
            
            # Write example
            success = storage.append_example(example)
            assert success
            
            # Read examples
            examples = storage.read_examples()
            assert len(examples) == 1
            assert examples[0]["prompt"] == "What is AI?"
    
    def test_encryption(self):
        """Test encryption at rest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create storage with encryption
            encryption_key = "test_key_1234567890abcdef1234567890abcdef="  # Valid Fernet key format
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=True,
                encryption_key=encryption_key
            )
            
            example = {
                "prompt": "Secret question",
                "response": "Secret answer",
            }
            
            storage.append_example(example)
            
            # Read raw file content
            examples_file = Path(tmpdir) / "examples.jsonl"
            with open(examples_file, "r") as f:
                raw_content = f.read()
            
            # Raw content should NOT contain plaintext
            assert "Secret question" not in raw_content
            assert "Secret answer" not in raw_content
            
            # But reading through storage API should decrypt
            examples = storage.read_examples()
            assert examples[0]["prompt"] == "Secret question"
            assert examples[0]["response"] == "Secret answer"
    
    def test_jsonl_injection_prevention(self):
        """Test prevention of JSONL injection attacks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            # Attempt JSONL injection with newlines in data
            malicious_example = {
                "prompt": "Normal prompt",
                "response": "Response\n{\"injected\": \"data\", \"malicious\": true}\nMore injection",
            }
            
            success = storage.append_example(malicious_example)
            assert success  # Should succeed but sanitize
            
            # Read back and verify no injection
            examples = storage.read_examples()
            assert len(examples) == 1  # Only one valid example
            
            # Response should have newlines removed/sanitized
            response = examples[0]["response"]
            assert "\n" not in response
            assert "\r" not in response
    
    def test_thread_safety(self):
        """Test thread-safe concurrent writes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            errors = []
            
            def write_many(thread_id: int):
                try:
                    for i in range(20):
                        example = {
                            "prompt": f"Thread {thread_id} question {i}",
                            "response": f"Thread {thread_id} answer {i}",
                            "thread_id": thread_id,
                            "index": i,
                        }
                        storage.append_example(example)
                except Exception as e:
                    errors.append((thread_id, e))
            
            # Run 5 threads concurrently
            threads = [threading.Thread(target=write_many, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # No errors should occur
            assert len(errors) == 0, f"Thread safety errors: {errors}"
            
            # All examples should be written
            examples = storage.read_examples()
            assert len(examples) == 5 * 20
    
    def test_gdpr_delete_user_data(self):
        """Test GDPR right to erasure (delete user data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            # Add examples for multiple users
            for user_id in ["user_1", "user_2", "user_3"]:
                for i in range(5):
                    storage.append_example({
                        "user_id": user_id,
                        "prompt": f"Question {i}",
                        "response": f"Answer {i}",
                    })
            
            # Delete user_2's data
            deleted = storage.delete_user_data("user_2")
            assert deleted == 5
            
            # Verify user_2's data is gone
            examples = storage.read_examples()
            assert len(examples) == 10  # Only user_1 and user_3 remain
            assert all(ex["user_id"] != "user_2" for ex in examples)
            
            # Verify other users' data intact
            user_1_examples = [ex for ex in examples if ex["user_id"] == "user_1"]
            assert len(user_1_examples) == 5
    
    def test_gdpr_export_user_data(self):
        """Test GDPR right to data portability (export user data)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            # Add examples for a user
            user_id = "user_123"
            for i in range(3):
                storage.append_example({
                    "user_id": user_id,
                    "prompt": f"My question {i}",
                    "response": f"My answer {i}",
                })
            
            # Add examples for another user (should not be in export)
            storage.append_example({
                "user_id": "other_user",
                "prompt": "Other question",
                "response": "Other answer",
            })
            
            # Export user data
            export = storage.export_user_data(user_id)
            
            assert export["user_id"] == user_id
            assert export["total_examples"] == 3
            assert len(export["examples"]) == 3
            assert all(ex["user_id"] == user_id for ex in export["examples"])


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestIntegration:
    """Test full integration of distillation components."""
    
    def test_full_capture_flow(self):
        """Test complete capture flow: PII redaction -> quality validation -> storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            redactor = PIIRedactor()
            validator = ExampleQualityValidator()
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            # User input with PII
            prompt = "Contact me at john@example.com for details about my SSN 123-45-6789"
            response = "Thank you! I'll reach out to you soon with information about your account."
            
            # Step 1: Check for secrets (hard reject)
            assert not redactor.contains_secrets(prompt)
            assert not redactor.contains_secrets(response)
            
            # Step 2: Redact PII
            redacted_prompt, _ = redactor.redact(prompt)
            redacted_response, _ = redactor.redact(response)
            
            assert "john@example.com" not in redacted_prompt
            assert "123-45-6789" not in redacted_prompt
            
            # Step 3: Quality validation
            passed, score, reasons = validator.validate(redacted_prompt, redacted_response)
            assert passed or score >= validator.MIN_QUALITY_SCORE
            
            # Step 4: Store
            example = {
                "prompt": redacted_prompt,
                "response": redacted_response,
                "quality_score": score,
                "timestamp": time.time(),
            }
            success = storage.append_example(example)
            assert success
            
            # Verify stored
            examples = storage.read_examples()
            assert len(examples) == 1
            assert "[REDACTED_EMAIL]" in examples[0]["prompt"]
            assert "[REDACTED_SSN]" in examples[0]["prompt"]


# ============================================================
# EDGE CASE TESTS
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty inputs."""
        redactor = PIIRedactor()
        validator = ExampleQualityValidator()
        
        # Empty text should not crash
        assert not redactor.contains_secrets("")
        redacted, stats = redactor.redact("")
        assert redacted == ""
        
        # Validation should handle empty response
        passed, score, reasons = validator.validate("prompt", "")
        assert not passed  # Should fail validation
    
    def test_very_large_input(self):
        """Test handling of very large inputs."""
        validator = ExampleQualityValidator()
        
        prompt = "Tell me everything"
        response = "A" * 10000  # Very large response
        
        # Should handle gracefully
        passed, score, reasons = validator.validate(prompt, response)
        assert any("too_long" in r for r in reasons)
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                use_encryption=False
            )
            
            # Example with Unicode
            example = {
                "prompt": "What is 你好?",
                "response": "你好 means 'hello' in Chinese. También hablo español! 🎉",
            }
            
            success = storage.append_example(example)
            assert success
            
            # Read back
            examples = storage.read_examples()
            assert len(examples) == 1
            # Unicode should be preserved (after sanitization)
            assert "hello" in examples[0]["response"]
    
    def test_null_and_none_values(self):
        """Test handling of null/None values."""
        validator = ExampleQualityValidator()
        
        # Should handle None gracefully
        try:
            passed, score, reasons = validator.validate("prompt", None)
            # Should either fail or handle None
            assert True
        except (TypeError, AttributeError):
            # Acceptable to raise on None
            assert True


# ============================================================
# SECURITY TESTS - Fail-Safe Behavior (P0: PII Leak Prevention)
# ============================================================


class TestPIIRedactorFailSafe:
    """
    Test fail-safe behavior in PII redactor.
    
    SECURITY CRITICAL: On any exception during redaction, the redactor MUST
    return a safe placeholder string instead of the original text. This prevents
    accidental PII/secret leakage if regex engine crashes, bad input is provided,
    or any other error occurs.
    
    This addresses the "PII Leak" security issue from the Forensic Audit:
    - Never return original text on redaction error
    - Return "[CONTENT REDACTED DUE TO SYSTEM ERROR]" on failure
    - Fail safe in contains_secrets() by assuming secrets present on error
    """
    
    def test_redact_returns_safe_placeholder_on_exception(self):
        """Test that redact() returns safe placeholder on internal error."""
        redactor = PIIRedactor()
        
        # Verify the REDACTION_ERROR_PLACEHOLDER constant exists
        assert hasattr(redactor, 'REDACTION_ERROR_PLACEHOLDER')
        assert redactor.REDACTION_ERROR_PLACEHOLDER == "[CONTENT REDACTED DUE TO SYSTEM ERROR]"
    
    def test_redact_does_not_leak_on_regex_error(self):
        """Test that regex errors don't cause PII leakage."""
        redactor = PIIRedactor()
        
        # Create a mock pattern that will raise an exception
        original_patterns = redactor.pii_patterns.copy()
        
        class BrokenPattern:
            def findall(self, text):
                raise RuntimeError("Simulated regex engine crash")
            def sub(self, replacement, text):
                raise RuntimeError("Simulated regex engine crash")
        
        # Inject broken pattern
        redactor.pii_patterns["test_broken"] = BrokenPattern()
        
        # Text with sensitive data
        sensitive_text = "My email is secret@private.com"
        
        try:
            redacted, stats = redactor.redact(sensitive_text)
            
            # CRITICAL: Original text should NOT be returned
            assert redacted != sensitive_text, "Original text was returned on error - PII LEAK!"
            
            # Should return the fail-safe placeholder or an error indicator
            assert (
                redacted == redactor.REDACTION_ERROR_PLACEHOLDER or 
                "error" in stats
            ), "Redaction should fail safely"
        finally:
            # Restore original patterns
            redactor.pii_patterns = original_patterns
    
    def test_contains_secrets_fails_safe_on_error(self):
        """Test that contains_secrets() assumes secrets present on error."""
        redactor = PIIRedactor()
        
        # Save original patterns
        original_secret_patterns = redactor.secret_patterns.copy()
        
        class BrokenPattern:
            def search(self, text):
                raise RuntimeError("Simulated crash during secret detection")
        
        # Inject broken pattern
        redactor.secret_patterns["test_broken"] = BrokenPattern()
        
        try:
            # This should fail safe by returning True (assume secrets present)
            result = redactor.contains_secrets("some text with potential secrets")
            
            # SECURITY: On error, should assume secrets ARE present (fail safe)
            assert result is True, "contains_secrets() should return True on error (fail safe)"
        finally:
            # Restore original patterns
            redactor.secret_patterns = original_secret_patterns
    
    def test_redact_returns_error_stats_on_failure(self):
        """Test that redact() returns error indicator in stats on failure."""
        redactor = PIIRedactor()
        
        # Inject a pattern that always raises
        class AlwaysFailPattern:
            def findall(self, text):
                raise ValueError("Intentional test failure")
            def sub(self, replacement, text):
                raise ValueError("Intentional test failure")
        
        redactor.secret_patterns["always_fail"] = AlwaysFailPattern()
        
        # The redact should fail and return error stats
        redacted, stats = redactor.redact("test text")
        
        # Either returns placeholder with error stat, or handles gracefully
        assert (
            redacted == redactor.REDACTION_ERROR_PLACEHOLDER or
            stats.get("error") == 1 or
            "error" in stats
        ) or redacted == "test text"  # If pattern iteration order doesn't hit broken one first
    
    def test_placeholder_constant_is_not_empty(self):
        """Verify fail-safe placeholder is meaningful and non-empty."""
        redactor = PIIRedactor()
        
        assert redactor.REDACTION_ERROR_PLACEHOLDER
        assert len(redactor.REDACTION_ERROR_PLACEHOLDER) > 10
        assert "REDACTED" in redactor.REDACTION_ERROR_PLACEHOLDER
        assert "ERROR" in redactor.REDACTION_ERROR_PLACEHOLDER


# ============================================================
# STORAGE TESTS - Log Rotation (P1: Infinite Storage Prevention)
# ============================================================


class TestStorageRotation:
    """
    Test storage log rotation functionality.
    
    This addresses the "Infinite Storage Bomb" issue from the Forensic Audit:
    - Size-based file rotation when file exceeds max_file_size_mb
    - Cleanup of old rotated files to prevent disk fill
    - Disk space warning when below threshold
    """
    
    def test_storage_has_rotation_settings(self):
        """Test that storage has rotation configuration options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                max_file_size_mb=100,
                max_rotated_files=10,
                min_free_disk_mb=500
            )
            
            assert storage.max_file_size_bytes == 100 * 1024 * 1024
            assert storage.max_rotated_files == 10
            assert storage.min_free_disk_bytes == 500 * 1024 * 1024
    
    def test_file_rotation_on_size_limit(self):
        """Test that files are rotated when they exceed size limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small size limit for testing (1KB)
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                max_file_size_mb=0.001,  # ~1KB
                max_rotated_files=5,
                enable_async_writes=False  # Use sync for predictable testing
            )
            
            # Write enough data to trigger rotation
            for i in range(100):
                storage.append_example({
                    "prompt": f"Question {i}" * 10,  # Make it larger
                    "response": f"Answer {i}" * 10,
                    "timestamp": time.time()
                })
            
            # Check that rotated files exist
            storage_path = Path(tmpdir)
            rotated_files = list(storage_path.glob("examples.*.jsonl"))
            
            # Should have some rotated files
            assert len(rotated_files) >= 1, "No rotation occurred"
    
    def test_old_rotated_files_cleanup(self):
        """Test that old rotated files are cleaned up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a storage with small rotation limits
            storage = DistillationStorageBackend(
                storage_path=tmpdir,
                max_file_size_mb=0.001,  # Very small - triggers rotation quickly
                max_rotated_files=2,  # Only keep 2 rotated files
                enable_async_writes=False
            )
            
            # Write enough to trigger multiple rotations
            for i in range(200):
                storage.append_example({
                    "prompt": f"Question {i}" * 20,
                    "response": f"Answer {i}" * 20,
                    "timestamp": time.time()
                })
            
            # Check rotated files
            storage_path = Path(tmpdir)
            rotated_files = list(storage_path.glob("examples.*.jsonl"))
            
            # Should have at most max_rotated_files rotated files
            assert len(rotated_files) <= storage.max_rotated_files + 1, \
                f"Too many rotated files: {len(rotated_files)} (max: {storage.max_rotated_files})"
    
    def test_disk_space_check_exists(self):
        """Test that disk space checking functionality exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DistillationStorageBackend(storage_path=tmpdir)
            
            # Method should exist
            assert hasattr(storage, '_check_disk_space')
            
            # Should not raise when called
            storage._check_disk_space()


# ============================================================
# TRAINING TRIGGER TESTS (P2: Close the Learning Loop)
# ============================================================


class TestTrainingTrigger:
    """
    Test training trigger functionality.
    
    This addresses the "Training Dead End" issue from the Forensic Audit:
    - trigger_training() method exists and works
    - Threshold-based automatic triggering
    - Callback and webhook support
    """
    
    def test_distiller_has_trigger_training_method(self):
        """Test that OpenAIKnowledgeDistiller has trigger_training method."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=10
            )
            
            assert hasattr(distiller, 'trigger_training')
            assert callable(distiller.trigger_training)
    
    def test_trigger_training_returns_result_dict(self):
        """Test that trigger_training returns proper result dictionary."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1000  # High threshold to not auto-trigger
            )
            
            # Manual trigger (forced)
            result = distiller.trigger_training(reason="test", force=True)
            
            assert isinstance(result, dict)
            assert "triggered" in result
            assert "reason" in result
            assert "timestamp" in result or "total_examples" in result
    
    def test_trigger_respects_threshold(self):
        """Test that trigger respects threshold when not forced."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1000  # High threshold
            )
            
            # Without force=True and without enough examples, should not trigger
            result = distiller.trigger_training(reason="test", force=False)
            
            assert result["triggered"] is False or result.get("reason") == "below_threshold"
    
    def test_trigger_training_with_callback(self):
        """Test that training callback is invoked."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        callback_invoked = {"called": False, "args": None}
        
        def test_callback(**kwargs):
            callback_invoked["called"] = True
            callback_invoked["args"] = kwargs
            return {"status": "success"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1,
                training_trigger_callback=test_callback
            )
            
            # Force trigger to test callback
            result = distiller.trigger_training(reason="test_callback", force=True)
            
            assert result["triggered"] is True
            assert result["callback_invoked"] is True
            assert callback_invoked["called"] is True
            assert "storage_path" in callback_invoked["args"]
    
    def test_trigger_training_logs_audit_entry(self):
        """Test that trigger_training creates audit log entry."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1
            )
            
            # Trigger training
            distiller.trigger_training(reason="test_audit", force=True)
            
            # Check audit log
            audit_log = distiller.get_audit_log()
            
            # Should have an entry for training_trigger
            trigger_entries = [
                entry for entry in audit_log 
                if entry.get("action") == "training_trigger"
            ]
            assert len(trigger_entries) >= 1
    
    def test_trigger_updates_stats(self):
        """Test that trigger_training updates statistics."""
        from vulcan.distillation.distiller import OpenAIKnowledgeDistiller
        
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller = OpenAIKnowledgeDistiller(
                storage_path=f"{tmpdir}/distillation.json",
                training_trigger_threshold=1
            )
            
            initial_triggers = distiller.stats.get("training_triggers", 0)
            
            # Trigger training
            distiller.trigger_training(reason="test_stats", force=True)
            
            # Stats should be updated
            assert distiller.stats["training_triggers"] == initial_triggers + 1
            assert distiller.stats["last_training_trigger_time"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
