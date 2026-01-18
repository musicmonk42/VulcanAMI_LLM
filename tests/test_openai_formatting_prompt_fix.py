"""
Test suite for OpenAI formatting prompt fix.

This test validates that the formatting prompt properly instructs OpenAI to:
1. Extract and present the conclusion field from nested JSON
2. Present the conclusion as the primary answer
3. Include supporting details (proof, explanation, reasoning_steps)
4. Never respond with just meta-commentary
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIFormattingPromptFix:
    """Test the OpenAI formatting prompt fix for proper conclusion presentation."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulate OpenAI returning a properly formatted response with actual conclusion
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="NO. The set is unsatisfiable. Proof: 1. From ¬C: C = False..."
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def executor_with_openai(self, mock_openai_client):
        """Create executor with mocked OpenAI."""
        from vulcan.llm.hybrid_executor import HybridLLMExecutor
        
        executor = HybridLLMExecutor(
            local_llm=MagicMock(),
            openai_client_getter=lambda: mock_openai_client,
            mode="parallel",
        )
        return executor

    @pytest.mark.asyncio
    async def test_sat_solver_conclusion_formatting(self, executor_with_openai, mock_openai_client):
        """Test that SAT solver conclusions are properly formatted with actual answer."""
        from vulcan.llm.hybrid_executor import VulcanReasoningOutput
        
        # Mock OpenAI to verify the prompt includes critical instructions
        captured_prompts = []
        
        def capture_create(*args, **kwargs):
            # Capture the user prompt from the messages
            messages = kwargs.get('messages', [])
            for msg in messages:
                if msg.get('role') == 'user':
                    captured_prompts.append(msg.get('content'))
            
            # Return a mock response with actual conclusion
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content="NO. The set is unsatisfiable. Proof: 1. From ¬C: C = False\n2. From B→C: B must be False\n3. From A∨B: A must be True\n4. But A→B forces B to be True\n5. Contradiction - the set is unsatisfiable."
                    )
                )
            ]
            return mock_response
        
        mock_openai_client.chat.completions.create = capture_create
        
        # Create a SAT solver output with nested conclusion
        reasoning_output = VulcanReasoningOutput(
            query_id="sat-001",
            success=True,
            result={
                "satisfiable": False,
                "result": "NO",
                "proof": "1. From ¬C: C = False\n2. From B→C: B must be False\n3. From A∨B: A must be True\n4. But A→B forces B to be True\n5. Contradiction",
                "conclusion": "The set is unsatisfiable"
            },
            result_type="symbolic",
            method_used="sat_solver",
            confidence=0.95,
        )
        
        # Format with OpenAI
        loop = asyncio.get_event_loop()
        formatted = await executor_with_openai._format_with_openai_for_output(
            reasoning_output=reasoning_output,
            original_query="Is A→B, B→C, ¬C, A∨B satisfiable?",
            loop=loop,
        )
        
        assert formatted is not None, "Formatting should succeed"
        
        # Verify the captured prompt contains critical instructions
        assert len(captured_prompts) > 0, "Should have captured at least one prompt"
        user_prompt = captured_prompts[0]
        
        # Verify critical instructions are in the prompt
        assert "conclusion" in user_prompt.lower(), "Prompt must mention 'conclusion'"
        assert "CRITICAL INSTRUCTIONS" in user_prompt, "Prompt must have CRITICAL INSTRUCTIONS header"
        assert "NEVER respond with just" in user_prompt, "Prompt must forbid meta-commentary"
        assert "Start your response with the actual answer" in user_prompt, "Prompt must instruct to start with answer"
        
        # Verify the response contains the actual answer
        assert "NO" in formatted or "unsatisfiable" in formatted.lower(), \
            f"Response should contain actual answer 'NO' or 'unsatisfiable', got: {formatted}"
        
        # Verify the response is NOT just meta-commentary
        assert not formatted.startswith("VULCAN successfully processed"), \
            "Response should not start with meta-commentary"
        assert "confidence" not in formatted or "proof" in formatted.lower(), \
            "If confidence mentioned, must also include substantive content like proof"

    @pytest.mark.asyncio
    async def test_mathematical_result_formatting(self, executor_with_openai, mock_openai_client):
        """Test that mathematical results are properly formatted with actual result."""
        from vulcan.llm.hybrid_executor import VulcanReasoningOutput
        
        # Mock OpenAI to return actual result
        mock_openai_client.chat.completions.create.return_value.choices[0].message.content = \
            "P(X|+) = 0.167 (or approximately 1/6)"
        
        reasoning_output = VulcanReasoningOutput(
            query_id="prob-001",
            success=True,
            result={
                "probability": 0.166667,
                "conclusion": "P(X|+) = 0.167",
                "explanation": "Using Bayes' theorem with given priors"
            },
            result_type="probabilistic",
            method_used="bayesian_inference",
            confidence=0.90,
        )
        
        loop = asyncio.get_event_loop()
        formatted = await executor_with_openai._format_with_openai_for_output(
            reasoning_output=reasoning_output,
            original_query="What is P(X|+)?",
            loop=loop,
        )
        
        assert formatted is not None
        # Should contain the actual probability value
        assert "0.167" in formatted or "1/6" in formatted, \
            f"Response should contain actual result, got: {formatted}"
        # Should NOT be just meta-commentary
        assert not (formatted.startswith("VULCAN") and "confidence" in formatted and "0.167" not in formatted), \
            "Should not be just meta-commentary without the actual result"

    @pytest.mark.asyncio
    async def test_prompt_structure_validation(self, executor_with_openai):
        """Test that the prompt has the correct structure with numbered instructions."""
        from vulcan.llm.hybrid_executor import VulcanReasoningOutput
        
        # Create a simple reasoning output
        reasoning_output = VulcanReasoningOutput(
            query_id="test-001",
            success=True,
            result={"answer": 42},
            confidence=0.95,
        )
        
        # Capture the prompt by mocking _call_openai_formatting
        captured_prompt = None
        
        async def mock_call_openai_formatting(loop, prompt, system_prompt, max_tokens, temperature):
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Test response"
        
        executor_with_openai._call_openai_formatting = mock_call_openai_formatting
        
        loop = asyncio.get_event_loop()
        await executor_with_openai._format_with_openai_for_output(
            reasoning_output=reasoning_output,
            original_query="Test query",
            loop=loop,
        )
        
        assert captured_prompt is not None, "Should have captured the prompt"
        
        # Verify structure
        assert "CRITICAL INSTRUCTIONS:" in captured_prompt, "Missing header"
        assert "1." in captured_prompt, "Missing instruction 1"
        assert "2." in captured_prompt, "Missing instruction 2"
        assert "3." in captured_prompt, "Missing instruction 3"
        assert "4." in captured_prompt, "Missing instruction 4"
        assert "5." in captured_prompt, "Missing instruction 5"
        assert "6." in captured_prompt, "Missing instruction 6"
        
        # Verify key instructions
        assert "conclusion" in captured_prompt.lower(), "Must mention finding conclusion"
        assert "proof" in captured_prompt.lower(), "Must mention including proof"
        assert "NEVER" in captured_prompt, "Must explicitly forbid bad patterns"

    def test_prompt_content_validation(self):
        """Test that the actual prompt in the code matches requirements."""
        import re
        from pathlib import Path
        
        # Use robust path resolution
        test_file_path = Path(__file__).resolve()
        project_root = test_file_path.parent.parent
        hybrid_executor_path = project_root / 'src' / 'vulcan' / 'llm' / 'hybrid_executor.py'
        
        assert hybrid_executor_path.exists(), f"Could not find hybrid_executor.py at {hybrid_executor_path}"
        
        # Read the actual code
        with open(hybrid_executor_path, 'r') as f:
            content = f.read()
        
        # Find the _format_with_openai_for_output method
        method_pattern = r'async def _format_with_openai_for_output\(.*?\):'
        method_match = re.search(method_pattern, content, re.DOTALL)
        assert method_match is not None, "Could not find _format_with_openai_for_output method"
        
        # Extract the method body (next 500 lines after method definition)
        method_start = method_match.end()
        method_section = content[method_start:method_start + 20000]
        
        # Find user_prompt in the method section
        # Look for the specific prompt that should contain our fix
        prompt_pattern = r'user_prompt = f"""(.*?)"""'
        matches = re.findall(prompt_pattern, method_section, re.DOTALL)
        
        # Find the formatting prompt (should be the first one in the method)
        target_prompt = None
        for match in matches:
            if "CRITICAL INSTRUCTIONS" in match or "formatting VULCAN's reasoning output" in match:
                target_prompt = match
                break
        
        assert target_prompt is not None, "Could not find the formatting prompt in _format_with_openai_for_output"
        
        # Verify all required elements
        required_keywords = [
            "conclusion",  # Must instruct to find conclusion
            "answer",      # Must mention presenting answer
            "proof",       # Must mention including proof/explanation
            "NEVER",       # Must forbid bad patterns
            "confidence",  # Must warn against "processed with confidence X"
            "Start",       # Must instruct to start with actual answer
        ]
        
        for keyword in required_keywords:
            assert keyword.lower() in target_prompt.lower(), \
                f"Prompt missing required keyword: {keyword}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
