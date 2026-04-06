"""
Symbolic Tool Wrapper Helpers - Preprocessing logic for symbolic reasoning queries.

Contains query preprocessing, header skipping, formal logic extraction, and
natural language to formal logic conversion.

Extracted from tool_selector.py to reduce module size.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SymbolicPreprocessingMixin:
    """
    Mixin providing query preprocessing methods for SymbolicToolWrapper.

    Transforms natural language queries into formal logic notation to enable
    the underlying SymbolicReasoner to process them correctly.
    """

    def _preprocess_query(self, query: str) -> str:
        """
        FIX #1: Preprocess natural language queries into formal logic notation.

        This addresses the core issue where engines expect formal logic but receive
        natural language, resulting in confidence=0.0.

        Transformations:
        1. Skip header/metadata lines that don't contain formal content
        2. Extract formal logic statements from mixed natural language/formal queries
        3. Normalize logical operators
        4. Handle SAT-style queries ("Is A->B, B->C satisfiable?")
        5. Handle FOL queries with quantifiers

        Args:
            query: Natural language or mixed query string

        Returns:
            Extracted/normalized formal logic string, or original if no formal content found

        Examples:
            "Is A->B, B->C, not C, A or B satisfiable?" -> "A->B, B->C, not C, A or B"
            "Symbolic Reasoning\\nS1 -- Satisfiability...\\n\\nPropositions: A,B,C..."
            -> "A->B, B->C, not C, A or B" (extracts the formal part)
        """
        if not query:
            return query

        original_query = query

        # ====================================================================
        # Skip header/metadata lines FIRST before other processing.
        # This prevents the bug where "Language Reasoning" header is parsed
        # instead of the actual SAT content below it.
        # ====================================================================
        cleaned_query = self._skip_header_lines(query)
        if cleaned_query != query:
            logger.info(
                f"[SymbolicEngine] Skipped header lines: "
                f"'{query[:30]}...' -> '{cleaned_query[:30]}...'"
            )
            query = cleaned_query

        # Step 1: Check if query already contains formal logic operators
        formal_operators = ['→', '∧', '∨', '¬', '∀', '∃', '->', '/\\', '\\/', '~', '⇒', '⇔', '|-']
        has_formal_content = any(op in query for op in formal_operators)

        if has_formal_content:
            # Try to extract just the formal logic portion
            extracted = self._extract_formal_logic_portion(query)
            if extracted:
                logger.info(f"[SymbolicEngine] Preprocessed query: '{query[:50]}...' -> '{extracted[:50]}...'")
                return extracted

        # Step 2: Check for natural language patterns that indicate logic queries
        # and try to convert them
        converted = self._convert_natural_language_to_formal(query)
        if converted != query:
            logger.info(f"[SymbolicEngine] Converted NL to formal: '{query[:50]}...' -> '{converted[:50]}...'")
            return converted

        # Step 3: Return original if no transformation needed/possible
        return original_query

    def _skip_header_lines(self, query: str) -> str:
        """
        Skip header/metadata lines from the query.

        The issue is that queries like:
            "Symbolic Reasoning
             S1 -- Satisfiability (SAT-style)

             Propositions: A, B, C
             Constraints:
             1. A->B
             ..."

        Were being parsed as just the header "Symbolic Reasoning" or "Language Reasoning",
        causing parse errors like "Unexpected token 'Reasoning'".

        This method skips:
        - Lines containing 'Reasoning' (headers like "Symbolic Reasoning", "Language Reasoning")
        - Lines with section markers like '\u2014' or '\u2013'
        - Lines starting with 'Task:', 'Claim:', 'S1', 'S2', etc.
        - Empty lines at the start

        Returns content starting from 'Propositions:', 'Constraints:', 'Formula:', etc.
        or the first line with actual formal content (logical operators).

        Args:
            query: Raw query string

        Returns:
            Query with header lines stripped, or original if no headers detected
        """
        if not query or '\n' not in query:
            return query

        lines = query.split('\n')
        content_lines = []
        found_content_start = False

        # Headers/metadata patterns to skip
        header_patterns = [
            'reasoning',  # "Symbolic Reasoning", "Language Reasoning"
            '\u2014',     # Section markers like "S1 \u2014 Satisfiability"
            '\u2013',     # Alternative dash
        ]

        # Content start markers (keep these lines)
        content_markers = [
            'proposition',  # "Propositions: A, B, C"
            'constraint',   # "Constraints:"
            'formula',      # "Formula:"
            'given',        # "Given:"
            'prove',        # "Prove:"
            'variables:',   # "Variables:"
            'task:',        # "Task: Is it satisfiable?" - this is content, not a header
            'claim:',       # "Claim:" - this is content, not a header
        ]

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Skip empty lines before content starts
            if not line_stripped and not found_content_start:
                continue

            # Check if this is a header line to skip
            is_header = False
            for pattern in header_patterns:
                if pattern in line_lower:
                    is_header = True
                    break

            # Check for S1, S2, M1, M2 style section markers at start of line
            if not is_header and line_stripped:
                # Match patterns like "S1", "S1 \u2014", "M2", etc. at line start
                if re.match(r'^[A-Z]\d+\s*[\u2014\u2013-]?\s*', line_stripped):
                    is_header = True

            # Check if this line starts content
            for marker in content_markers:
                if marker in line_lower:
                    found_content_start = True
                    break

            # Check if line contains formal logic operators (content line)
            formal_operators = ['→', '∧', '∨', '¬', '∀', '∃', '->', '|-']
            if any(op in line_stripped for op in formal_operators):
                found_content_start = True

            # Keep the line if it's content or we've found content
            if found_content_start or not is_header:
                content_lines.append(line)
                if line_stripped:  # Mark that we found content
                    found_content_start = True

        # Return cleaned content, or original if no content found
        cleaned = '\n'.join(content_lines).strip()
        return cleaned if cleaned else query

    def _extract_formal_logic_portion(self, query: str) -> Optional[str]:
        """
        Extract formal logic statements from a query that contains both
        natural language and formal notation.

        Example:
            Input: "Is A->B, B->C, not C, A or B satisfiable?"
            Output: "A->B, B->C, not C, A or B"
        """
        # Pattern 1: Look for comma-separated formulas with operators
        formula_pattern = r'([A-Z∀∃¬→∧∨⇒⇔()a-z_\s,~]+(?:→|∧|∨|¬|⇒|⇔|->)[A-Z∀∃¬→∧∨⇒⇔()a-z_\s,~]+)'

        matches = re.findall(formula_pattern, query)
        if matches:
            longest = max(matches, key=len)
            cleaned = ' '.join(longest.split())
            if len(cleaned) > 3:
                return cleaned

        # Pattern 2: Look for explicit formula sections
        if "Proposition" in query or "Formula" in query:
            lines = query.split('\n')
            formula_lines = []
            for line in lines:
                line = line.strip()
                if any(skip in line.lower() for skip in ['symbolic', 'reasoning', 'satisfiability', 'step', 'analyze']):
                    continue
                if any(op in line for op in ['→', '∧', '∨', '¬', '->', '|', '&']):
                    formula_lines.append(line)
            if formula_lines:
                return ', '.join(formula_lines)

        # Pattern 3: Extract from parenthesized expressions
        paren_pattern = r'\(([^()]+(?:→|∧|∨|¬|->)[^()]+)\)'
        paren_matches = re.findall(paren_pattern, query)
        if paren_matches:
            return ', '.join(paren_matches)

        return None

    def _convert_natural_language_to_formal(self, query: str) -> str:
        """
        Convert natural language logic queries to formal notation.

        Handles patterns like:
        - "if A then B" -> "A -> B"
        - "A and B" -> "A and B"
        - "A or B" -> "A or B"
        - "not A" -> "not A"
        - "for all X" -> "for all X"
        - "there exists X" -> "there exists X"
        """
        result = query

        # Natural language to formal operator mappings
        replacements = [
            (r'\bif\s+(\w+)\s+then\s+(\w+)\b', r'\1 → \2'),
            (r'\b(\w+)\s+implies\s+(\w+)\b', r'\1 → \2'),
            (r'\b(\w+)\s+if\s+and\s+only\s+if\s+(\w+)\b', r'\1 ⇔ \2'),
            (r'\b(\w+)\s+iff\s+(\w+)\b', r'\1 ⇔ \2'),
            (r'\b(\w+)\s+and\s+(\w+)\b', r'\1 ∧ \2'),
            (r'\b(\w+)\s+or\s+(\w+)\b', r'\1 ∨ \2'),
            (r'\bnot\s+([A-Z])\b', r'¬\1'),
            (r'\bfor\s+all\s+(\w+)\b', r'∀\1'),
            (r'\bthere\s+exists\s+(\w+)\b', r'∃\1'),
            (r'\bexists\s+(\w+)\b', r'∃\1'),
        ]

        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result
